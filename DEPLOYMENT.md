# Deployment Guide — DigitalOcean Droplet (Bangalore)

Ye guide maan kar chalti hai ki aapne pehle kabhi deploy nahi kiya. Har command copy-paste
karne layak hai. Jahan aapko kuch **decide** karna hai ya kuch **likhna** hai, wahan saaf
likha gaya hai.

**Architecture:**

```
Internet
   │
   ▼  HTTPS (port 443)
┌──────────────── Droplet (BLR1) ─────────────────┐
│  nginx    TLS + reverse proxy                    │
│  certbot  certificate renew karta rehta hai      │
│  api      calls uthata hai, agent chalata hai    │
│  worker   lead extract karta hai                 │
│  redis    cache + job queue + counters           │
│  migrate  ek baar chalta hai, phir band          │
└──────────────────────────────────────────────────┘
   │  private network
   ▼
DigitalOcean Managed Postgres (BLR1)
```

Postgres droplet par **nahi** chalega — wo aapka managed cluster hai. Redis droplet par hi
chalega, container mein.

---

## 0. Shuru karne se pehle kya chahiye

| Cheez | Kahan se |
|---|---|
| DigitalOcean account (billing on) | cloud.digitalocean.com |
| Managed Postgres cluster, BLR1 | aapke paas already hai |
| Domain `homebble.in` ka DNS access | aapka domain registrar |
| GitHub repo access | `markanthony-projects/AI_CALLING_AGENT` |
| Vobiz dashboard login | webhook URL set karne ke liye |
| Provider keys | Groq, Deepgram, Sarvam, OpenAI — already hain |
| SSH key aapke laptop par | Step 1 mein bana lenge |

**Time:** pehli baar mein 60–90 minute. Jaldi mat kijiye.

---

## 1. SSH key banaiye (laptop par)

Ye aapke laptop ka "password" hai jisse droplet mein ghusenge. Password se zyada safe hai.

PowerShell mein:

```powershell
ssh-keygen -t ed25519 -C "homebble-deploy"
```

Teen baar Enter dabaiye (default location, khaali passphrase — ya passphrase daal dijiye,
zyada safe hai). Ab public key dekhiye:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Jo lamba `ssh-ed25519 AAAA...` string dikhe, wo **copy** kar lijiye. Agli step mein chahiye.

> `id_ed25519` (bina `.pub`) aapki **private** key hai. Ye kisi ko kabhi mat dijiye.

---

## 2. Droplet banaiye

DigitalOcean console → **Create → Droplets**

| Setting | Kya chunein | Kyun |
|---|---|---|
| Region | **Bangalore (BLR1)** | DB bhi BLR1 mein hai — same region zaroori |
| OS | **Ubuntu 24.04 LTS x64** | long-term support |
| Type | **Premium Intel**, 4 vCPU / 8 GB | neeche padhiye |
| Authentication | **SSH Key** → naya add karein, Step 1 wali key paste karein | password login se safe |
| VPC Network | **wahi VPC jisme aapka Postgres hai** | private network se DB connect hoga |
| Hostname | `homebble-voice-prod` | pehchanne ke liye |
| Monitoring | ✅ enable | free hai |

**Size kyun 4 vCPU?** App ek waqt mein 8 calls tak leti hai (`MAX_CALLS = 8`), aur har call
mein Silero VAD CPU par chalta hai plus audio resampling hota hai. 2 vCPU par 8 calls mein
audio tootne lagega. Agar shuru mein sirf 1–2 test calls karni hain to 2 vCPU / 4 GB se
shuru kar sakte hain, baad mein resize ho jaata hai (resize ke liye reboot lagta hai).

Create dabaiye. 1 minute mein IP address mil jaayega — **usko note kar lijiye.**

---

## 3. DNS record banaiye

Apne domain registrar mein (jahan `homebble.in` khareeda hai):

| Field | Value |
|---|---|
| Type | `A` |
| Name / Host | `ai-calls` |
| Value / Points to | droplet ka IP |
| TTL | 300 (ya Automatic) |

Save kijiye. Ab check kijiye ki phaila ya nahi (laptop par):

```powershell
nslookup ai-calls.homebble.in
```

Jab droplet ka IP dikhne lage, tabhi aage badhiye. **5–30 minute lag sakte hain.**

> Ye step certificate lene se pehle hona **zaroori** hai. Let's Encrypt is naam par HTTP
> request bhejkar check karta hai ki server aapka hai. DNS nahi phaila to certificate fail
> hoga, aur production CA mein **ek ghante mein sirf 5 failed attempts** allowed hain.

---

## 4. Cloud firewall lagaiye

Console → **Networking → Firewalls → Create Firewall**

Name: `homebble-voice-fw`

**Inbound Rules** — pehle jo default hai use delete karke ye teen banaiye:

| Type | Protocol | Port | Sources |
|---|---|---|---|
| SSH | TCP | 22 | **My IP** (dropdown se) |
| HTTP | TCP | 80 | All IPv4, All IPv6 |
| HTTPS | TCP | 443 | All IPv4, All IPv6 |

**Outbound Rules:** default hi rehne dijiye (sab allowed).

Neeche **Apply to Droplets** mein apna droplet chunein. Create dabaiye.

**Port 80 kyun khula?** Let's Encrypt certificate issue aur renew karne ke liye port 80 par
challenge file maangta hai, aur nginx HTTP se HTTPS par redirect karta hai. Port 8000 kahin
nahi khulega — wo `expose` hai, `ports` nahi, matlab sirf container network ke andar dikhta
hai. Firewall galat hone par bhi internet se uspar route nahi hai.

---

## 5. Droplet mein login aur basic hardening

```powershell
ssh root@ai-calls.homebble.in
```

Pehli baar `yes` type karna padega. Ab andar ye sab chalaiye — **ek-ek line**:

```bash
# System update
apt-get update && apt-get upgrade -y

# Roz ke kaam ke liye non-root user (root se seedha kaam karna theek nahi)
adduser --disabled-password --gecos "" deploy
usermod -aG sudo deploy

# Apni SSH key us user ko bhi de dijiye
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
```

Ab root login band kar dijiye:

```bash
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sshd -t && systemctl restart ssh
```

> `sshd -t` config test karta hai. Agar wo error de to **restart mat kijiye** — warna
> aap khud bahar ho jaayenge. Error aaye to mujhe bataiye.

Automatic security updates:

```bash
apt-get install -y unattended-upgrades
dpkg-reconfigure -f noninteractive unattended-upgrades
```

Ab `exit` karke naye user se login kijiye:

```powershell
exit
ssh deploy@ai-calls.homebble.in
```

**Ab se saara kaam `deploy` user se hoga.**

---

## 6. Docker install kijiye

```bash
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# deploy user ko docker chalane ki permission
sudo usermod -aG docker deploy
```

Ab **logout-login** kijiye (group change tabhi lagta hai):

```bash
exit
```
```powershell
ssh deploy@ai-calls.homebble.in
```

Check:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Teeno chalein to Docker ready hai.

---

## 7. Managed Postgres ko droplet se jodiye

### 7a. Trusted source add kijiye

Console → **Databases → aapka cluster → Settings → Trusted Sources → Edit**

Apna **droplet** chunein (dropdown mein naam se dikhega). Save kijiye.

Agar pehle "All IPv4" tha to **use hata dijiye** — ab sirf droplet hi DB tak pahunche.

> Dhyaan: is ke baad aapke laptop se DB connect nahi hoga. Local development ke liye
> apna IP bhi add karna padega, ya laptop wale kaam ke liye alag entry rakhein.

### 7b. Private connection string lijiye

Usi page par **Connection Details** → dropdown mein **VPC network** chunein (Public network
nahi). `Connection string` copy kar lijiye. Wo aisa dikhega:

```
postgresql://doadmin:PASSWORD@private-db-blr1-xxxxx.b.db.ondigitalocean.com:25060/defaultdb?sslmode=require
```

Private network use karne se DB traffic internet par nahi jaata — tez bhi hai aur safe bhi.

### 7c. URL ko app ke format mein badliye

Do badlaav kijiye:

1. `?sslmode=require` ko **`?ssl=require`** kar dijiye — asyncpg `ssl` samajhta hai, `sslmode` nahi
2. Database ka naam wahi rakhiye jo aap use kar rahe hain (aapka current DB naam)

Final aisa hoga:

```
postgresql://doadmin:PASSWORD@private-db-blr1-xxxxx.b.db.ondigitalocean.com:25060/YOURDB?ssl=require
```

> **`ssl=require` kyun zaroori hai?** Agar ye na ho to asyncpg `ssl=prefer` use karta hai,
> jiska matlab: SSL try karega, fail hone par **chup-chaap plaintext par gir jaayega**, aur
> server ka certificate verify bhi nahi karega. `require` se SSL compulsory ho jaata hai.
> Maine ye aapke cluster par test kiya hai — `ssl=require` kaam karta hai.

---

## 8. Code droplet par laiye

Repo private hai, to GitHub **deploy key** banate hain (password se behtar).

```bash
ssh-keygen -t ed25519 -C "droplet-deploy-key" -f ~/.ssh/github_deploy -N ""
cat ~/.ssh/github_deploy.pub
```

Jo output aaye use copy kijiye. Ab GitHub par:

**Repo → Settings → Deploy keys → Add deploy key**
- Title: `homebble-voice-droplet`
- Key: paste kijiye
- **Allow write access: OFF** (droplet ko sirf padhna hai)

Add kijiye. Ab droplet par SSH config:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/github_deploy
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Clone kijiye:

```bash
cd ~
git clone git@github.com:markanthony-projects/AI_CALLING_AGENT.git app
cd app
```

Pehli baar `yes` type karna padega.

> **Agar clone fail ho:** matlab aapka local kaam abhi GitHub par push nahi hua. Laptop par
> `git status` dekhiye — bahut si files untracked hain. Push karna hai to mujhe bataiye,
> main commit aur push kar dunga.

---

## 9. `.env` file banaiye

**Ye sabse important step hai.** Yahan galti hui to kuch nahi chalega.

Pehle secrets banaiye (droplet par hi):

```bash
python3 scripts/gen_secrets.py
```

Output ko safe jagah note kar lijiye — screen par ek hi baar aayega.

Ab file banaiye:

```bash
nano .env
```

Neeche wala paste kijiye aur `<...>` waale hisse apne values se badliye:

```bash
# ---- Auth (production) ----
# AUTH_ENABLED yahan likhna hi nahi hai. Default true hai. Likhne se galti ka risk hai.
API_KEY=<gen_secrets se pehli line>
CALL_TOKEN_SECRET=<gen_secrets se doosri line>
CALL_TOKEN_TTL_SECONDS=900
DOCS_ENABLED=false

# ---- Database (managed, Step 7c) ----
DATABASE_URL=postgresql://doadmin:<PASSWORD>@<private-host>:25060/<DBNAME>?ssl=require

# ---- Redis: compose isko override karta hai, phir bhi rakhiye ----
REDIS_URL=redis://redis:6379/0

# ---- AI providers ----
DEEPGRAM_API_KEY=<aapki key>
SARVAM_API_KEY=<aapki key>
SARVAM_VOICE_ID=simran
OPENAI_API_KEY=<aapki key>
GROQ_API_KEY=<aapki key>

# ---- Telephony ----
VOBIZ_AUTH_ID=<aapka id>
VOBIZ_AUTH_TOKEN=<aapka token>
VOBIZ_PHONE_NUMBER=<aapka number>
WEBHOOK_BASE_URL=https://ai-calls.homebble.in

# ---- Dial ceilings (leaked key ka nuksaan seemit karta hai) ----
DIAL_MAX_PER_MINUTE=30
DIAL_MAX_PER_DAY=500

DEFAULT_COUNTRY_CODE=91
```

Save: `Ctrl+O` → Enter → `Ctrl+X`

Ab permission lock kijiye:

```bash
chmod 600 .env
ls -l .env      # -rw------- dikhna chahiye
```

**`WEBHOOK_BASE_URL` mein trailing slash mat dijiye.** `https://ai-calls.homebble.in` — bas.

---

## 10. Stack start kijiye

### 10a. Pehle app containers start kijiye (nginx ke bina)

```bash
docker compose -f docker-compose.prod.yml up -d --build redis migrate api worker
```

Pehli baar 3–6 minute lagenge (Docker image ban raha hai).

```bash
docker compose -f docker-compose.prod.yml ps
```

| Service | State |
|---|---|
| `redis` | Up (healthy) |
| `migrate` | **Exited (0)** ← ye sahi hai |
| `api` | Up |
| `worker` | Up |

> `migrate` ka `Exited (0)` **success** hai. Usne DB migration chala kar kaam khatam kiya.
> `Exited (1)` ho to migration fail hui — `docker compose -f docker-compose.prod.yml logs migrate`

### 10b. TLS certificate lijiye

Nginx bina certificate ke start nahi hota, aur certbot ko certificate lene ke liye nginx
chahiye. Ye murgi-anda problem ek script solve karti hai — wo pehle ek nakli certificate
rakhti hai, nginx start karti hai, phir asli certificate laakar nakli hata deti hai.

**Pehle staging par test kijiye** (production CA mein ghanta bhar mein sirf 5 galtiyaan
allowed hain):

```bash
STAGING=1 LETSENCRYPT_EMAIL=you@homebble.in ./scripts/init_letsencrypt.sh
```

Script pehle check karti hai ki `ai-calls.homebble.in` isi droplet par point karta hai. Agar
DNS galat hai to wo wahin ruk jaayegi — ye jaan-boojh kar hai, warna aap rate limit mein
phas jaate.

`curl -k https://ai-calls.homebble.in/health` chale to staging kaam kar gaya. Ab asli
certificate lijiye:

```bash
rm -rf ./certbot/conf/live ./certbot/conf/archive ./certbot/conf/renewal
LETSENCRYPT_EMAIL=you@homebble.in ./scripts/init_letsencrypt.sh
```

### 10c. Poora stack start kijiye

```bash
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

Ab saat services mein se ye dikhni chahiye:

| Service | State |
|---|---|
| `redis` | Up (healthy) |
| `migrate` | Exited (0) |
| `api` | Up |
| `worker` | Up |
| `nginx` | Up |
| `certbot` | Up |

`certbot` background mein chalta rehta hai — har 12 ghante jaagta hai aur 30 din se kam
bache certificate ko renew karta hai. Nginx har 6 ghante khud reload karta hai, to naya
certificate apne aap uth jaata hai.

---

## 11. Sab kuch verify kijiye

### 11a. HTTPS aur auth

Laptop se:

```powershell
curl https://ai-calls.homebble.in/health
```

Expected:

```json
{"status":"ok","auth":"enabled"}
```

- `"auth":"enabled"` — auth chalu hai ✅
- Browser mein padlock dikhna chahiye

Agar certificate ka problem ho to:

```bash
docker compose -f docker-compose.prod.yml logs nginx
docker compose -f docker-compose.prod.yml exec nginx nginx -t
docker compose -f docker-compose.prod.yml run --rm --entrypoint "certbot certificates" certbot
```

Aakhri command batayegi ki certificate kis din expire hoga.

### 11b. Docs band hain

```powershell
curl -o /dev/null -w "%{http_code}" https://ai-calls.homebble.in/docs
```

**404** aana chahiye. Aa gaya to schema public nahi hai ✅

### 11c. API key ke bina block hota hai

```powershell
curl -o /dev/null -w "%{http_code}" -X POST https://ai-calls.homebble.in/api/v1/campaigns/
```

**401** aana chahiye ✅

### 11d. DB migration ho gayi

```bash
docker compose -f docker-compose.prod.yml exec api alembic current
```

`db0682dd0be0 (head)` dikhna chahiye.

### 11e. Worker zinda hai

```bash
docker compose -f docker-compose.prod.yml logs --tail 20 worker
```

Har minute ye line aani chahiye:

```
recording health: j_complete=0 j_failed=0 j_ongoing=0 queued=0
```

`queued=0` matlab koi kaam pending nahi ✅

---

## 12. Vobiz ko naya URL bataiye

Vobiz dashboard mein aapka answer URL abhi pinggy tunnel par hai. Usko badal dijiye:

```
https://ai-calls.homebble.in/vobiz/answer/{campaign_id}/{call_sid}
```

App khud hi `?token=...` जोड़ता hai jab call place karta hai, to aapko token manually nahi
dena. Bas base URL sahi hona chahiye.

---

## 13. Pehli test call

Apne hi number par kijiye — kisi customer par nahi.

```powershell
curl -X POST "https://ai-calls.homebble.in/api/v1/campaigns/cdd5da76-a87d-4f98-9f32-5fe1f7869b77/dial/vobiz" `
  -H "X-API-Key: <aapka API_KEY>" `
  -H "Content-Type: application/json" `
  -d '{\"phone_numbers\":[\"+919604100447\"]}'
```

Saath hi droplet par logs dekhte rahiye:

```bash
docker compose -f docker-compose.prod.yml logs -f api worker
```

**Kya dekhna hai:**

| Check | Expected |
|---|---|
| Priya pehle bolti hai | 1–2 second mein, silence nahi |
| Beech mein rukein | turn na toote |
| Din bataayein, time nahi | agent time poochhe, call band na kare |
| Time bhi bataayein | sign-off mein time repeat ho |
| Log mein | `phone=+91...`, `unit=...`, `status=...` |

Phir DB check:

```bash
docker compose -f docker-compose.prod.yml exec api python -c "
import asyncio
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.db import Lead
async def m():
    async with AsyncSessionLocal() as db:
        l = (await db.execute(select(Lead).order_by(Lead.created_at.desc()).limit(1))).scalars().first()
        for f in ('customer_name','phone_number','budget','preferred_unit_type','status','site_visit_time'):
            print(f'{f:20} = {getattr(l,f)}')
asyncio.run(m())
"
```

---

## 14. Roz ke kaam (Operations)

```bash
cd ~/app

# Logs
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f worker
docker compose -f docker-compose.prod.yml logs --tail 100 nginx
docker compose -f docker-compose.prod.yml logs --tail 50 certbot

# Restart
docker compose -f docker-compose.prod.yml restart api

# Naya code deploy
git pull
docker compose -f docker-compose.prod.yml up -d --build

# Status
docker compose -f docker-compose.prod.yml ps
docker stats --no-stream
```

### ⚠️ Kabhi ye mat chalaiye

```bash
docker compose -f docker-compose.prod.yml down -v     # ❌ -v Redis ka data uda dega
```

`-v` volumes delete karta hai. Redis mein pending extraction jobs hoti hain — wo gum ho
jaayengi aur un calls ki leads kabhi nahi banengi. Sirf `down` (bina `-v`) safe hai.

### Test data saaf karna (jab production shuru karein)

```bash
docker compose -f docker-compose.prod.yml exec api python -c "
import asyncio
from sqlalchemy import text
from app.core.database import engine
async def m():
    async with engine.begin() as c:
        await c.execute(text('TRUNCATE transcripts, calls, leads RESTART IDENTITY CASCADE'))
        print('cleared')
asyncio.run(m())
"
```

Projects aur campaigns bache rahenge — agent unhi se pitch karta hai.

---

## 15. Kuch galat ho to

| Problem | Kya karein |
|---|---|
| `curl /health` timeout | Cloud firewall mein 443 khula hai? `docker compose ... ps` mein nginx Up hai? |
| Certificate nahi mila | `nslookup` sahi IP de raha hai? Port 80 khula hai? `logs certbot` |
| Deploy ke baad `502 Bad Gateway` | `logs nginx` dekhiye. Config mein resolver hai to apne aap theek ho jaata hai 10s mein; warna `restart nginx` |
| `nginx: [emerg] cannot load certificate` | certificate nahi bana. Step 10b dobara chalaiye |
| `migrate` Exited (1) | `logs migrate` — 99% cases mein `DATABASE_URL` galat hai ya trusted source add nahi hua |
| DB connection refused | Step 7a: droplet trusted sources mein hai? Private host use kiya? |
| `401` sahi key ke saath bhi | header `X-API-Key` hai (case matter karta hai)? `.env` mein extra space? |
| Call connect hoti hai, awaaz nahi | `logs api` mein Sarvam/Deepgram error dekhiye — credits khatam? |
| Lead nahi ban rahi | `logs worker`. `queued` badhta ja raha hai? Worker crash hua? |
| Call ke beech audio toot rahi | droplet chhota hai — `docker stats` mein CPU dekhiye, resize karein |

Logs bhejne ke liye:

```bash
docker compose -f docker-compose.prod.yml logs --tail 200 api > /tmp/api.log
```

---

## 16. Go-live se pehle final checklist

- [ ] `curl /health` → `{"status":"ok","auth":"enabled"}`
- [ ] `/docs` → 404
- [ ] Bina API key → 401
- [ ] Browser mein padlock (valid TLS)
- [ ] `alembic current` → `db0682dd0be0 (head)`
- [ ] Cloud firewall: sirf 22 (My IP), 80, 443
- [ ] DB trusted sources: sirf droplet
- [ ] `DATABASE_URL` mein `?ssl=require` hai
- [ ] `.env` permissions `600`
- [ ] Root SSH login band
- [ ] Vobiz webhook naye domain par
- [ ] Vobiz account par **balance alert** laga hua
- [ ] Ek test call safaltapoorvak, lead DB mein
- [ ] Test data truncate kiya

---

## Baaki cheezein jo abhi nahi hain

Ye jaan-boojh kar chhodi gayi hain, chhupayi nahi gayi:

- **Database backup** — DO managed Postgres daily backup deta hai, par aapne verify nahi
  kiya ki restore chalta hai. Ek baar test restore kar lena chahiye.
- **`doadmin` superuser** — app DB ka admin user use kar rahi hai. Production grade mein
  ek limited user hona chahiye jo sirf apni tables padh/likh sake.
- **`ssl=verify-full`** — abhi `require` hai. `verify-full` server ka certificate bhi
  verify karta hai, par uske liye DO ka CA certificate file droplet par rakhni padegi.
- **Ek hi droplet** — droplet gaya to sab gaya. Uptime chahiye to load balancer + do
  droplets, par tab Redis ko shared karna padega.
- **Alerting** — abhi kuch nahi hai jo aapko batayega ki calls fail ho rahi hain. Logs
  khud dekhne padenge.
- **Read APIs** — sales team ke liye leads dekhne ka koi endpoint nahi hai.

Inme se koi bhi chahiye to bataiye.
