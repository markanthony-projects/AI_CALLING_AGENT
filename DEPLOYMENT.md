# Deployment Guide

DigitalOcean Droplet, Bangalore. Written for someone deploying for the first time — every
command can be copied as-is. Where you have to decide something or type your own value, it
is marked clearly.

## Architecture

```
Internet
   │
   ▼  HTTPS :443
┌──────────── Droplet (BLR1, 2 vCPU / 4 GB) ─────────────┐
│  nginx      terminates TLS, proxies to api             │
│  certbot    renews the certificate in the background   │
│  api        answers calls, runs the voice agent        │
│  worker     extracts leads after each call             │
│  redis      cache + job queue + dial counters          │
│  migrate    runs once at startup, then exits           │
└────────────────────────────────────────────────────────┘
   │  TLS over the public internet, restricted by Trusted Sources
   ▼
DigitalOcean Managed Postgres (BLR1)
```

Postgres does **not** run on the droplet — it is your managed cluster. Redis **does** run on
the droplet, as a container. Redis is not only a cache: the job queue lives in it, and the
API and worker are separate processes that cannot pass work to each other any other way.

**Capacity:** **3 concurrent calls**. Two limits meet here and the lower one wins. The CPU
allows about four — each call runs voice activity detection and audio resampling, roughly two
calls per vCPU before audio breaks up — but the Vobiz account permits three simultaneous
calls, so three is the ceiling. It also leaves CPU for the API, Redis, the worker and nginx.

Raising `MAX_CONCURRENT_CALLS` above the carrier's limit does not buy throughput: the extra
dials are placed, billed, and refused by Vobiz.

---

## 0. What you need before starting

| Item | Where it comes from |
|---|---|
| DigitalOcean account with billing enabled | cloud.digitalocean.com |
| Managed Postgres cluster in BLR1 | you already have this |
| DNS access for `homebble.in` | your domain registrar |
| GitHub repo access | `markanthony-projects/AI_CALLING_AGENT` |
| Vobiz dashboard login | to update the webhook URL |
| Provider API keys | Groq, Deepgram, Sarvam, OpenAI — you already have these |
| An email address | Let's Encrypt sends expiry warnings to it |

**Time:** 60–90 minutes the first time. Do not rush.

---

## 1. Create the droplet

DigitalOcean console → **Create → Droplets**

| Setting | Choose | Why |
|---|---|---|
| Region | **Bangalore (BLR1)** | same region as your database |
| OS | **Ubuntu 24.04 LTS x64** | long-term support |
| Type | **Regular**, 2 vCPU / 4 GB | carries 3 concurrent calls with headroom |
| Authentication | **Password** | see below |
| Hostname | `homebble-voice-prod` | so you can identify it later |
| Monitoring | enable | free |

### About password authentication

You asked for password login so that you or a colleague can get in from any machine without
carrying a key file. That is a reasonable trade for a small team, but be clear about what it
costs: **automated bots scan port 22 continuously and try common passwords.** SSH keys cannot
be guessed; passwords can.

So the password itself has to do the work. Generate one — do not invent it by hand:

```powershell
python -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24)))"
```

Paste that as the root password when creating the droplet, and store it in your company
password manager (1Password, Bitwarden, Keeper — whatever your team already uses). A 24
character random password is not guessable in any practical timeframe.

In **Step 5** you will install `fail2ban`, which bans an IP after repeated failed logins.
Password authentication without fail2ban is genuinely risky; with it, it is acceptable.

Press Create. You get an IP address in about a minute — **write it down.**

---

## 2. Create the DNS record

In your domain registrar's DNS panel:

| Field | Value |
|---|---|
| Type | `A` |
| Name / Host | `ai-calls` |
| Value / Points to | your droplet's IP |
| TTL | 300 or Automatic |

Save, then check from your laptop:

```powershell
nslookup ai-calls.homebble.in
```

Wait until it returns your droplet's IP before continuing. This takes **5–30 minutes.**

> This must be working before you request a certificate. Let's Encrypt proves you own the
> domain by making an HTTP request to it. If DNS is not ready the request fails, and the
> production service allows only **5 failures per hour per domain.**

---

## 3. Create the cloud firewall

Console → **Networking → Firewalls → Create Firewall**

Name: `homebble-voice-fw`

**Inbound rules** — delete the defaults and create these three:

| Type | Protocol | Port | Sources |
|---|---|---|---|
| SSH | TCP | 22 | All IPv4, All IPv6 |
| HTTP | TCP | 80 | All IPv4, All IPv6 |
| HTTPS | TCP | 443 | All IPv4, All IPv6 |

**Outbound rules:** leave the defaults.

Under **Apply to Droplets**, select your droplet. Press Create.

> **If your office has a fixed IP address**, set the SSH source to that IP instead of "All".
> This is the single most effective protection available and costs nothing. You would then
> only need to widen it when working from elsewhere.

**Why is port 80 open?** Let's Encrypt requires it to issue and renew certificates, and
nginx uses it to redirect visitors to HTTPS. Port 8000 is never exposed — it is declared
`expose` rather than `ports` in the compose file, so it exists only on the internal container
network. Even a misconfigured firewall cannot route to it.

---

## 4. Log in

```powershell
ssh root@ai-calls.homebble.in
```

Type `yes` the first time, then your password.

---

## 5. Secure the server

You are working as `root`. That is what you asked for, and it is workable — but root has no
safety net, so the protections below are not optional.

### 5a. Update the system

```bash
apt-get update && apt-get upgrade -y
```

### 5b. Install fail2ban

This is the piece that makes password login safe. It watches the SSH log and bans any IP
that fails repeatedly.

```bash
apt-get install -y fail2ban

cat > /etc/fail2ban/jail.local <<'EOF'
[sshd]
enabled  = true
port     = ssh
backend  = systemd
maxretry = 4
findtime = 10m
bantime  = 24h
EOF

systemctl enable --now fail2ban
```

Verify it is running:

```bash
fail2ban-client status sshd
```

You should see a `Currently banned` count. Within a day or two it will not be zero — that is
bots being turned away, and it is the reason this step matters.

### 5c. Tighten SSH

```bash
cat > /etc/ssh/sshd_config.d/99-hardening.conf <<'EOF'
PermitRootLogin yes
PasswordAuthentication yes
MaxAuthTries 3
LoginGraceTime 30
X11Forwarding no
ClientAliveInterval 300
ClientAliveCountMax 2
EOF

sshd -t && systemctl restart ssh
```

> `sshd -t` tests the configuration. **If it reports an error, do not restart** — you would
> lock yourself out. Fix the error first, or ask for help while your current session is still
> open.

`MaxAuthTries 3` means three wrong passwords per connection; fail2ban then bans the IP after
four such connections.

### 5d. Enable automatic security updates

```bash
apt-get install -y unattended-upgrades
dpkg-reconfigure -f noninteractive unattended-upgrades
```

### 5e. Keep a recovery route

If you ever get locked out — wrong password, or fail2ban bans your own office IP — use the
**DigitalOcean console** (droplet page → **Console** button, top right). It opens a terminal
in your browser that does not go through SSH, so it works even when SSH does not. To clear a
self-inflicted ban:

```bash
fail2ban-client set sshd unbanip YOUR.IP.HERE
```

---

## 6. Install Docker

```bash
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Check all three work:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

---

## 7. Connect to the managed database

### 7a. Allow the droplet through

Console → **Databases → your cluster → Settings → Trusted Sources → Edit**

Select your **droplet** from the dropdown. Save.

If "All IPv4" was listed before, **remove it.** Trusted Sources is a firewall on the database
itself — with only the droplet listed, nothing else on the internet can even attempt to
connect, regardless of whether it has the password.

> After this your laptop can no longer reach the database directly. If you need that for
> local development, add your own IP as a second entry.

### 7b. Copy the connection string

Same page → **Connection Details** → **Connection string** → copy it.

It looks like:

```
postgresql://doadmin:PASSWORD@db-blr1-xxxxx.b.db.ondigitalocean.com:25060/defaultdb?sslmode=require
```

### 7c. Adjust it for this application

Make one change: replace `?sslmode=require` with **`?ssl=require`**.

```
postgresql://doadmin:PASSWORD@db-blr1-xxxxx.b.db.ondigitalocean.com:25060/defaultdb?ssl=require
```

Keep whatever database name you are already using in place of `defaultdb`.

> **Why the rename?** The driver this application uses (`asyncpg`) recognises `ssl`, not
> `sslmode`. Without it the driver defaults to "prefer", which tries SSL and **silently falls
> back to an unencrypted connection** if it fails, without verifying the server's certificate.
> `ssl=require` makes encryption mandatory. This was tested against your cluster.

### About VPC / private networking

You may have noticed a "VPC network" option in the connection details. Ignore it. Here is
what it does and why you do not need it:

A VPC keeps database traffic on DigitalOcean's internal network instead of the public
internet. That sounds important, but in your setup it adds little:

- The connection is already encrypted with TLS (`ssl=require`)
- The database already rejects everyone except your droplet (Trusted Sources)
- Both machines are in the same datacentre, so the latency difference is negligible
- Database queries do not happen during a call — project data is cached in Redis, and the
  call record is written after the call ends. So this is not on the latency path at all.

It is a defence-in-depth measure for larger deployments. **Use the public connection string
with `ssl=require`.** It is simpler, has fewer ways to go wrong, and is entirely appropriate
at your scale.

---

## 8. Get the code onto the droplet

The repository is private, so create a **deploy key** — a key that grants read-only access to
this one repository.

```bash
ssh-keygen -t ed25519 -C "droplet-deploy-key" -f ~/.ssh/github_deploy -N ""
cat ~/.ssh/github_deploy.pub
```

Copy the output. Then on GitHub:

**Repo → Settings → Deploy keys → Add deploy key**

- Title: `homebble-voice-droplet`
- Key: paste it
- **Allow write access: leave OFF** — the droplet only needs to read

Back on the droplet:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/github_deploy
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

cd ~
git clone git@github.com:markanthony-projects/AI_CALLING_AGENT.git app
cd app
```

Type `yes` at the host key prompt.

### 8b. The dashboard is not deployed here

The operations dashboard is a separate repository deployed to **Vercel**. Nothing about it
is built or served on this droplet — this box runs the backend only.

What the backend owes it is one setting: `DASHBOARD_CORS_ORIGINS` naming the dashboard's
origin, covered in Step 9.

**Where you host it on Vercel matters more than it looks.** The login session is a cookie
set by `ai-calls.homebble.in`, and whether the browser will send that cookie back depends on
whether the dashboard is the same *site*:

| Dashboard URL | Relationship | Result |
|---|---|---|
| `dashboard.homebble.in` (Vercel custom domain) | same site | Works everywhere. Keep `DASHBOARD_COOKIE_SAMESITE=lax` |
| `your-app.vercel.app` (default Vercel domain) | cross site | Needs `SameSite=None`, which makes the session a **third-party cookie** |

Safari and Firefox **block third-party cookies by default today**, and Chrome is heading the
same way. On the default `.vercel.app` domain, sign-in will simply not persist for those
users — the request succeeds, the cookie is discarded, and every call after it is anonymous.

**Add a custom domain in Vercel** (project → Settings → Domains → `dashboard.homebble.in`,
then the CNAME they give you at your registrar). Both hosts are then under `homebble.in`,
the cookie is first-party, and nothing has to be weakened.

---

## 9. Create the `.env` file

This is the step most worth double-checking. A mistake here means nothing starts.

First generate the two application secrets:

```bash
python3 scripts/gen_secrets.py
```

These are values you create, not values you obtain from anyone. Note them down — they are
printed once.

Now create the file:

```bash
nano .env
```

Paste the following and replace every `<...>` with your own value:

```bash
# ---- Authentication ----
# AUTH_ENABLED is deliberately absent. It defaults to true; writing it out only creates
# a chance of setting it to false by accident.
API_KEY=<first line from gen_secrets.py>
CALL_TOKEN_SECRET=<second line from gen_secrets.py>
CALL_TOKEN_TTL_SECONDS=900
DOCS_ENABLED=false

# ---- Operations dashboard ----
# Signs dashboard login sessions. Generate a THIRD secret, distinct from the two above:
#   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Leave empty to disable the dashboard — /api/v1/auth and /api/v1/dashboard then return 503.
DASHBOARD_SESSION_SECRET=<a third, different secret>
DASHBOARD_SESSION_TTL_SECONDS=43200
# The Vercel origin, exactly — scheme included, no trailing slash. Credentialed CORS
# forbids "*", so a wrong value here silently blocks every dashboard request.
DASHBOARD_CORS_ORIGINS=https://dashboard.homebble.in
DASHBOARD_COOKIE_SECURE=true
# lax if the dashboard is on a homebble.in subdomain (recommended). none only if it stays
# on a .vercel.app URL — and then Safari and Firefox will discard the session. See Step 8b.
DASHBOARD_COOKIE_SAMESITE=lax

# ---- Database (from Step 7c) ----
DATABASE_URL=postgresql://doadmin:<PASSWORD>@<your-db-host>:25060/<DBNAME>?ssl=require

# ---- Redis (compose overrides this; keep it anyway) ----
REDIS_URL=redis://redis:6379/0

# ---- AI providers ----
DEEPGRAM_API_KEY=<your key>
SARVAM_API_KEY=<your key>
SARVAM_VOICE_ID=simran
OPENAI_API_KEY=<your key>
GROQ_API_KEY=<your key>

# ---- Telephony ----
VOBIZ_AUTH_ID=<your id>
VOBIZ_AUTH_TOKEN=<your token>
VOBIZ_PHONE_NUMBER=<your number>
WEBHOOK_BASE_URL=https://ai-calls.homebble.in

# ---- Capacity and spend limits ----
MAX_CONCURRENT_CALLS=3
DIAL_MAX_PER_MINUTE=30
DIAL_MAX_PER_DAY=500

DEFAULT_COUNTRY_CODE=91
```

Save with `Ctrl+O`, Enter, then `Ctrl+X`.

Lock the file down:

```bash
chmod 600 .env
ls -l .env      # should show -rw-------
```

Two things to get right:

- **No trailing slash** on `WEBHOOK_BASE_URL`. Just `https://ai-calls.homebble.in`
- **No spaces** around the `=` signs

---

## 10. Start everything

Three stages, because nginx cannot start until a certificate exists.

### 10a. Start the application containers

```bash
docker compose -f docker-compose.prod.yml up -d --build redis migrate api worker
```

The first build takes 3–6 minutes. Then:

```bash
docker compose -f docker-compose.prod.yml ps
```

| Service | Expected state |
|---|---|
| `redis` | Up (healthy) |
| `migrate` | **Exited (0)** |
| `api` | Up |
| `worker` | Up |

> `Exited (0)` for `migrate` means **success.** It applied the database migrations and
> finished. `Exited (1)` means it failed — check
> `docker compose -f docker-compose.prod.yml logs migrate`. Almost always a wrong
> `DATABASE_URL` or a missing Trusted Sources entry.

### 10b. Obtain the TLS certificate

Nginx will not start without a certificate, and certbot cannot obtain one until nginx is
serving the challenge. A script resolves this: it installs a temporary self-signed
certificate, starts nginx, then replaces it with the real one.

**Test with the staging service first.** The production service allows only 5 failures per
hour, and a DNS mistake is much cheaper to discover here:

```bash
STAGING=1 LETSENCRYPT_EMAIL=you@homebble.in ./scripts/init_letsencrypt.sh
```

The script first verifies that `ai-calls.homebble.in` resolves to this droplet. If it does
not, it stops — deliberately, so you do not spend your rate limit on a known-bad setup.

Confirm it worked:

```bash
curl -k https://ai-calls.homebble.in/health
```

(`-k` skips certificate validation, which is expected with staging.)

Now get the real certificate:

```bash
rm -rf ./certbot/conf/live ./certbot/conf/archive ./certbot/conf/renewal
LETSENCRYPT_EMAIL=you@homebble.in ./scripts/init_letsencrypt.sh
```

### 10c. Start the full stack

```bash
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

You should now have six services, with `nginx` and `certbot` both Up.

`certbot` runs continuously — it wakes every 12 hours and renews any certificate within 30
days of expiry. Nginx reloads itself every 6 hours so a renewed certificate is picked up
without you doing anything.

---

## 11. Verify the deployment

### 11a. HTTPS and authentication

From your laptop:

```powershell
curl https://ai-calls.homebble.in/health
```

Expected:

```json
{"status":"ok","auth":"enabled"}
```

`"auth":"enabled"` confirms authentication is active. Your browser should also show a padlock
with no warning.

If the certificate is not working:

```bash
docker compose -f docker-compose.prod.yml logs nginx
docker compose -f docker-compose.prod.yml exec nginx nginx -t
docker compose -f docker-compose.prod.yml run --rm --entrypoint "certbot certificates" certbot
```

The last command shows the expiry date.

### 11b. The API schema is not public

From the droplet (bash):

```bash
curl -o /dev/null -s -w "%{http_code}
" https://ai-calls.homebble.in/docs
```

From your laptop (PowerShell), `/dev/null` becomes `NUL`:

```powershell
curl -o NUL -s -w "%{http_code}" https://ai-calls.homebble.in/docs
```

Expected: **404**

### 11c. Requests without a key are rejected

```bash
curl -o /dev/null -s -w "%{http_code}
" -X POST https://ai-calls.homebble.in/api/v1/campaigns/
```

Expected: **401**

### 11d. Database migrations applied

```bash
docker compose -f docker-compose.prod.yml exec api alembic current
```

Expected: `db0682dd0be0 (head)`

### 11e. The worker is alive

```bash
docker compose -f docker-compose.prod.yml logs --tail 20 worker
```

Every minute you should see:

```
recording health: j_complete=0 j_failed=0 j_ongoing=0 queued=0
```

`queued=0` means nothing is waiting. If that number grows and never falls, the worker has
stopped picking up jobs.

### 11f. fail2ban is watching

```bash
fail2ban-client status sshd
```

---

## 12. Update Vobiz

In the Vobiz dashboard, change the answer URL to:

```
https://ai-calls.homebble.in/vobiz/answer/{campaign_id}/{call_sid}
```

The application appends the `?token=...` itself when placing a call — you do not add it.

---

## 12b. Create a dashboard login

There is no sign-up page: this dashboard reads every prospect's phone number and every call
transcript, so accounts are provisioned here.

```bash
docker compose -f docker-compose.prod.yml exec api \
  python scripts/manage_dashboard_users.py create you@homebble.in --role ADMIN
```

It prompts for a password twice and never echoes it. Minimum 12 characters. Do **not** pass
it as an argument — argv lands in shell history and in the process list.

| Role | Can do |
|---|---|
| `VIEWER` | Read everything; reclassify a lead's temperature |
| `ADMIN` | The above, plus pause/resume campaigns and start dialing runs |

Give sales `VIEWER`. `ADMIN` can spend money, so keep it to the people who own the budget.

Other commands:

```bash
# ... exec api python scripts/manage_dashboard_users.py <command>
list                       # who has access, and when they last signed in
passwd you@homebble.in     # change a password
deactivate ex@homebble.in  # revoke access — takes effect on their next page load
```

Now open the **dashboard** URL from Step 8b — not this API host — and sign in.

> **Why not just use the API key?** Because it dials, and dialing costs money. A key pasted
> into a browser lives in devtools, in history, and in every extension on the page. The
> dashboard authenticates as a named person and carries a session in an httpOnly cookie that
> page script cannot read.

If the dashboard shows **503 "Dashboard is not configured"**, `DASHBOARD_SESSION_SECRET` is
unset or shorter than 32 characters. Fix `.env`, then `up -d` (a restart does not reload it).

If login appears to succeed but every page then says you are signed out, check
`DASHBOARD_COOKIE_SECURE=true` and that you are on **https**, not http.

---

## 13. Make a test call

Call your own number, not a customer's.

First find a campaign ID:

```bash
docker compose -f docker-compose.prod.yml exec api python -c "
import asyncio
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.db import Campaign
async def main():
    async with AsyncSessionLocal() as db:
        for c in (await db.execute(select(Campaign))).scalars().all():
            print(c.id, '|', c.name)
asyncio.run(main())
"
```

**On the droplet (bash)** — reads the key straight out of `.env`, so there is nothing to
mistype:

```bash
curl -X POST "https://ai-calls.homebble.in/api/v1/campaigns/<CAMPAIGN_ID>/dial/vobiz" \
  -H "X-API-Key: $(grep '^API_KEY=' .env | cut -d= -f2-)" \
  -H "Content-Type: application/json" \
  -d '{"phone_numbers":["+919604100447"]}'
```

**From your laptop (PowerShell)** — note the different line-continuation character and the
escaped quotes:

```powershell
curl -X POST "https://ai-calls.homebble.in/api/v1/campaigns/<CAMPAIGN_ID>/dial/vobiz" `
  -H "X-API-Key: <your API_KEY>" `
  -H "Content-Type: application/json" `
  -d '{\"phone_numbers\":[\"+919604100447\"]}'
```

> **Do not mix the two.** In PowerShell a backtick continues the line; in bash a backtick
> means "run this as a command", so a PowerShell-style command pasted into bash tries to
> execute the header as a program and the request goes out with no key at all. Bash uses a
> backslash. Inside bash single quotes, double quotes need no escaping.

Watch the logs while it runs:

```bash
docker compose -f docker-compose.prod.yml logs -f api worker
```

What to check during the call:

| Test | Expected behaviour |
|---|---|
| Call connects | Priya speaks within 1–2 seconds, no silence |
| Pause mid-sentence | your turn is not split; the agent waits |
| Give a day but no time | the agent asks for a time and does **not** hang up |
| Then give a time | the closing line repeats the day and time back to you |
| Worker log | `phone=+91...`, `unit=...`, `status=...` all populated |

Then confirm the data landed:

```bash
docker compose -f docker-compose.prod.yml exec api python -c "
import asyncio
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.db import Lead
async def main():
    async with AsyncSessionLocal() as db:
        lead = (await db.execute(
            select(Lead).order_by(Lead.created_at.desc()).limit(1)
        )).scalars().first()
        for f in ('customer_name','phone_number','budget','preferred_unit_type',
                  'status','site_visit_time'):
            print(f'{f:20} = {getattr(lead, f)}')
asyncio.run(main())
"
```

---

## 14. Testing after deployment

The server is live but you are still testing. A few things make that easier.

**Enable the API docs temporarily.** Swagger gives you an "Authorize" button so you can paste
the API key once and test every endpoint from the browser:

```bash
sed -i 's/^DOCS_ENABLED=.*/DOCS_ENABLED=true/' .env
docker compose -f docker-compose.prod.yml up -d api
```

Then open `https://ai-calls.homebble.in/docs`. **Set it back to `false` when you are done** —
it publishes your full route list.

**Test without spending Vobiz credit.** The browser test client runs the same voice agent
through your browser's microphone instead of the phone network:

```bash
curl -X POST "https://ai-calls.homebble.in/api/v1/campaigns/<CAMPAIGN_ID>/dial/browser"   -H "X-API-Key: $(grep '^API_KEY=' .env | cut -d= -f2-)"
```

It returns a link to open. Note that this still consumes Groq, Deepgram and Sarvam credit —
only the telephony leg is free.

**Clear test data before going live:**

```bash
docker compose -f docker-compose.prod.yml exec api python -c "
import asyncio
from sqlalchemy import text
from app.core.database import engine
async def main():
    async with engine.begin() as c:
        await c.execute(text('TRUNCATE transcripts, calls, leads RESTART IDENTITY CASCADE'))
        print('cleared')
asyncio.run(main())
"
```

Projects and campaigns are preserved — those are what the agent pitches from.

---

## 15. Day-to-day operations

```bash
cd ~/app

# Logs
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f worker
docker compose -f docker-compose.prod.yml logs --tail 100 nginx

# Restart one service
docker compose -f docker-compose.prod.yml restart api

# Apply a change made to .env  ← NOT the same as restart
docker compose -f docker-compose.prod.yml up -d --force-recreate api worker

# Deploy new code
git pull
docker compose -f docker-compose.prod.yml up -d --build

# Status and resource use
docker compose -f docker-compose.prod.yml ps
docker stats --no-stream
```

### Never run this

```bash
docker compose -f docker-compose.prod.yml down -v      # the -v is the problem
```

`-v` deletes volumes, including Redis. Any extraction jobs still queued are lost, and those
calls never produce a lead. Plain `down` without `-v` is safe.

### Raising capacity later

Two things have to move together, and the carrier is usually the binding one. Raise the Vobiz
concurrent-call limit first; without that, a higher cap here only places calls the carrier
will refuse — after billing the attempt.

Then resize the droplet to match, at roughly two calls per vCPU, and update the cap:

```bash
sed -i 's/^MAX_CONCURRENT_CALLS=.*/MAX_CONCURRENT_CALLS=<new limit>/' .env
docker compose -f docker-compose.prod.yml up -d --force-recreate api worker
```

Both containers, because the dial pump in the worker is what reserves slots, and
`--force-recreate` because `env_file` is read when a container is created — a plain `up -d`
can report "up to date" and leave the old value running.

Setting it higher than the host can carry degrades every call in progress instead of
rejecting the extra one.

---

## 16. Troubleshooting

| Symptom | What to check |
|---|---|
| `curl /health` times out | Is 443 open in the cloud firewall? Is `nginx` Up in `ps`? |
| Certificate not issued | Does `nslookup` return the right IP? Is port 80 open? `logs certbot` |
| `migrate` exited (1) | `logs migrate`. Nearly always a wrong `DATABASE_URL` or a missing Trusted Sources entry |
| Database connection refused | Step 7a — is the droplet in Trusted Sources? Is `?ssl=require` present? |
| `502 Bad Gateway` after deploying | `logs nginx`. Should self-correct within 10 seconds; if not, `restart nginx` |
| `401` even with the right key | Did you change `.env` and only `restart`? Environment variables are read when a container is **created**, so a running container keeps the old values. Compare `docker compose … exec -T api printenv API_KEY` against `.env`, then `up -d --force-recreate api worker` |
| `curl: (43) Failed sending HTTP POST request` | The header value contains a newline. Either `.env` has two `API_KEY=` lines, or it has Windows line endings. Check with `grep -c '^API_KEY=' .env` and `grep '^API_KEY=' .env \| cat -A` |
| Call connects but there is no audio | `logs api` — look for Sarvam or Deepgram errors. Out of credit? |
| No lead created | `logs worker`. Is `queued` climbing? Did the worker crash? |
| Audio breaking up | `docker stats` — if CPU is saturated, lower `MAX_CONCURRENT_CALLS` or resize |
| Locked out of SSH | Use the DigitalOcean browser console, then `fail2ban-client set sshd unbanip YOUR.IP` |
| Dashboard returns 503 | `DASHBOARD_SESSION_SECRET` is unset or under 32 chars. Fix `.env` then `up -d` — a `restart` does not reload it |
| Dashboard login works, then every page says signed out | The cookie is `Secure`; you must be on **https**. Confirm `DASHBOARD_COOKIE_SECURE=true` and that you are not on `http://` |
| Dashboard 403 "Cross-origin request rejected" | The dashboard is on a different origin than the API without `DASHBOARD_CORS_ORIGINS` naming it. Same-host deployments should leave that variable empty |
| Dashboard shows the API's JSON instead of the UI | nginx routed the path to `api`. The API owns only `/api`, `/vobiz`, `/static`, `/health`, `/docs` — check the `location` regex in `nginx/conf.d/app.conf` |
| Dashboard login succeeds but every later request is 401 | The session cookie is being discarded. On a `.vercel.app` URL it is a third-party cookie and Safari/Firefox drop it — move the dashboard to a `homebble.in` subdomain (Step 8b). Also check `DASHBOARD_COOKIE_SECURE=true`, since a Secure cookie is never stored over plain HTTP |
| Dashboard requests blocked by CORS in the browser console | `DASHBOARD_CORS_ORIGINS` must match the dashboard's origin exactly — scheme included, no trailing slash |
| "Live now" shows 0 during a real call | `active_calls` is counted per API process. With one `api` replica it is exact; the dial-quota meters beside it come from Redis and hold across replicas |

To capture logs for sharing:

```bash
docker compose -f docker-compose.prod.yml logs --tail 200 api > /tmp/api.log
```

---

## 17. Final checklist before going live

- [ ] `curl /health` returns `{"status":"ok","auth":"enabled"}`
- [ ] `/docs` returns 404
- [ ] A request without an API key returns 401
- [ ] Browser shows a valid padlock
- [ ] `alembic current` shows `db0682dd0be0 (head)`
- [ ] `fail2ban-client status sshd` reports the jail is active
- [ ] Cloud firewall allows only 22, 80, 443
- [ ] Database Trusted Sources lists only the droplet
- [ ] `DATABASE_URL` contains `?ssl=require`
- [ ] `.env` permissions are `600`
- [ ] Root password is 20+ random characters and stored in a password manager
- [ ] Vobiz points at the new domain
- [ ] A **balance alert** is configured on your Vobiz account
- [ ] One successful test call, with the lead visible in the database
- [ ] Test data cleared

---

## What this deployment does not yet include

Listed openly rather than left to be discovered:

- **Verified backups.** DigitalOcean backs up managed Postgres daily, but you have not tested
  a restore. Untested backups are not backups.
- **A limited database user.** The application connects as `doadmin`, the cluster
  administrator. A dedicated user with access only to its own tables would be better.
- **Per-person server accounts.** Everyone shares the root password, so the logs cannot tell
  you who did what. Fine for two or three people who trust each other; worth revisiting as
  the team grows.
- **Certificate pinning for the database.** `ssl=require` encrypts the connection but does not
  verify the server's certificate. `ssl=verify-full` would, but needs DigitalOcean's CA
  certificate placed on the droplet.
- **A second server.** One droplet means one point of failure. Redundancy needs a load
  balancer and shared Redis.
- **Alerting.** Nothing notifies you when calls start failing. You have to read the logs.
- **Read APIs for the sales team.** There is no endpoint yet for viewing leads.

Ask when you want any of these.
