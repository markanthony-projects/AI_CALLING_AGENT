"""Create, list, and manage dashboard logins.

There is no sign-up page on purpose: this dashboard reads every prospect's phone number and
every call transcript, so accounts are provisioned by whoever administers the box.

    python scripts/manage_dashboard_users.py create ops@homebble.in --role ADMIN
    python scripts/manage_dashboard_users.py list
    python scripts/manage_dashboard_users.py passwd ops@homebble.in
    python scripts/manage_dashboard_users.py deactivate ops@homebble.in

The password is read from a prompt that does not echo, or from DASHBOARD_USER_PASSWORD for
unattended provisioning. It is never taken as an argument — argv lands in shell history and
in the process list, where every other user on the box can read it.
"""

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.core.database import AsyncSessionLocal, engine
from app.core.passwords import hash_password
from app.models.db import DashboardRole, DashboardUser

MIN_PASSWORD_LENGTH = 12


def _read_password(confirm: bool = True) -> str:
    from_env = os.environ.get("DASHBOARD_USER_PASSWORD")
    if from_env:
        password = from_env
    else:
        password = getpass.getpass("Password: ")
        if confirm and password != getpass.getpass("Confirm password: "):
            raise SystemExit("Passwords do not match.")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise SystemExit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    return password


async def _create(email: str, role: str, full_name: str | None) -> None:
    email = email.strip().lower()
    async with AsyncSessionLocal() as db:
        existing = (
            await db.execute(select(DashboardUser).where(DashboardUser.email == email))
        ).scalars().first()
        if existing:
            raise SystemExit(f"{email} already exists. Use 'passwd' to change the password.")

        password = _read_password()
        db.add(
            DashboardUser(
                email=email,
                full_name=full_name,
                password_hash=hash_password(password),
                role=DashboardRole(role),
            )
        )
        await db.commit()
    print(f"Created {email} with role {role}.")


async def _passwd(email: str) -> None:
    email = email.strip().lower()
    async with AsyncSessionLocal() as db:
        user = (
            await db.execute(select(DashboardUser).where(DashboardUser.email == email))
        ).scalars().first()
        if not user:
            raise SystemExit(f"No such user: {email}")
        user.password_hash = hash_password(_read_password())
        await db.commit()
    print(f"Password updated for {email}. Existing sessions stay valid until they expire.")


async def _set_active(email: str, active: bool) -> None:
    email = email.strip().lower()
    async with AsyncSessionLocal() as db:
        user = (
            await db.execute(select(DashboardUser).where(DashboardUser.email == email))
        ).scalars().first()
        if not user:
            raise SystemExit(f"No such user: {email}")
        user.is_active = active
        await db.commit()
    # /auth/me re-reads the row on every dashboard load, so a deactivated user is locked
    # out on their next navigation rather than when their token happens to expire.
    print(f"{email} is now {'active' if active else 'deactivated'}.")


async def _list() -> None:
    async with AsyncSessionLocal() as db:
        users = (
            await db.execute(select(DashboardUser).order_by(DashboardUser.email))
        ).scalars().all()
    if not users:
        print("No dashboard users yet. Create one with the 'create' command.")
        return
    print(f"{'EMAIL':<36} {'ROLE':<8} {'ACTIVE':<7} LAST LOGIN")
    for u in users:
        last = u.last_login_at.strftime("%Y-%m-%d %H:%M") if u.last_login_at else "never"
        print(f"{u.email:<36} {u.role.value:<8} {str(u.is_active):<7} {last}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a new dashboard user")
    create.add_argument("email")
    create.add_argument("--role", choices=[r.value for r in DashboardRole], default=DashboardRole.VIEWER.value)
    create.add_argument("--name", dest="full_name", default=None)

    passwd = sub.add_parser("passwd", help="Change a user's password")
    passwd.add_argument("email")

    deactivate = sub.add_parser("deactivate", help="Revoke a user's access")
    deactivate.add_argument("email")

    activate = sub.add_parser("activate", help="Restore a user's access")
    activate.add_argument("email")

    sub.add_parser("list", help="List dashboard users")

    args = parser.parse_args()
    try:
        if args.command == "create":
            await _create(args.email, args.role, args.full_name)
        elif args.command == "passwd":
            await _passwd(args.email)
        elif args.command == "deactivate":
            await _set_active(args.email, False)
        elif args.command == "activate":
            await _set_active(args.email, True)
        elif args.command == "list":
            await _list()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
