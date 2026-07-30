#!/bin/bash
# First-time certificate issue for the Nginx service.
#
# Chicken and egg: nginx refuses to start without the files named in ssl_certificate, and
# certbot cannot answer the HTTP challenge until nginx is serving. So this puts a throwaway
# self-signed certificate at the expected path, starts nginx with it, then replaces it with
# a real one and reloads.
#
# Run once, from the repo root on the droplet:
#     LETSENCRYPT_EMAIL=you@homebble.in ./scripts/init_letsencrypt.sh
#
# Set STAGING=1 to use Let's Encrypt's staging CA while testing. Production has a rate limit
# of 5 failed attempts per hour per domain; staging does not, and a browser warning is a
# cheaper way to find a DNS mistake than a lockout.

set -euo pipefail

DOMAIN="ai-calls.homebble.in"
EMAIL="${LETSENCRYPT_EMAIL:?Set LETSENCRYPT_EMAIL=you@homebble.in}"
STAGING="${STAGING:-0}"

COMPOSE="docker compose -f docker-compose.prod.yml"
CONF="./certbot/conf"
LIVE="$CONF/live/$DOMAIN"

if [ -d "$LIVE" ]; then
	echo "A certificate for $DOMAIN already exists at $LIVE."
	echo "Renewal is automatic. To force a fresh one, delete that directory first."
	exit 0
fi

echo "==> Checking that $DOMAIN resolves to this droplet"
resolved="$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1 || true)"
public="$(curl -fsS --max-time 10 https://api.ipify.org || true)"
if [ -z "$resolved" ]; then
	echo "FAIL: $DOMAIN does not resolve. Create the DNS A record and wait for it to spread."
	exit 1
fi
if [ -n "$public" ] && [ "$resolved" != "$public" ]; then
	echo "FAIL: $DOMAIN resolves to $resolved but this droplet is $public."
	echo "Fix the DNS A record, or the challenge will be answered by the wrong host."
	exit 1
fi
echo "    $DOMAIN -> $resolved (matches this droplet)"

echo "==> Fetching recommended TLS settings"
mkdir -p "$CONF" "./certbot/www"
if [ ! -e "$CONF/options-ssl-nginx.conf" ]; then
	curl -fsS https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf \
		-o "$CONF/options-ssl-nginx.conf"
fi
if [ ! -e "$CONF/ssl-dhparams.pem" ]; then
	curl -fsS https://raw.githubusercontent.com/certbot/certbot/master/certbot/certbot/ssl-dhparams.pem \
		-o "$CONF/ssl-dhparams.pem"
fi

echo "==> Placing a throwaway certificate so nginx can boot"
mkdir -p "$LIVE"
$COMPOSE run --rm --entrypoint "\
	openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
		-keyout '/etc/letsencrypt/live/$DOMAIN/privkey.pem' \
		-out '/etc/letsencrypt/live/$DOMAIN/fullchain.pem' \
		-subj '/CN=localhost'" certbot

echo "==> Starting nginx"
$COMPOSE up -d nginx
sleep 5

echo "==> Removing the throwaway certificate"
$COMPOSE run --rm --entrypoint "\
	rm -rf /etc/letsencrypt/live/$DOMAIN \
		/etc/letsencrypt/archive/$DOMAIN \
		/etc/letsencrypt/renewal/$DOMAIN.conf" certbot

echo "==> Requesting the real certificate"
staging_arg=""
if [ "$STAGING" != "0" ]; then
	staging_arg="--staging"
	echo "    (staging CA — the browser will warn, that is expected)"
fi

$COMPOSE run --rm --entrypoint "\
	certbot certonly --webroot -w /var/www/certbot \
		$staging_arg \
		--email $EMAIL \
		-d $DOMAIN \
		--rsa-key-size 4096 \
		--agree-tos \
		--no-eff-email \
		--non-interactive" certbot

echo "==> Reloading nginx"
$COMPOSE exec nginx nginx -s reload

echo
echo "Done. Verify with:"
echo "    curl https://$DOMAIN/health"
