"""
Seed Script — populate Postgres with realistic mock doctors and slots.

Run once after running Alembic migrations (or init_db):
    python scripts/seed_db.py

Creates:
  - 15 doctors across key specialties
  - 30+ appointment slots per doctor over the next 7 days
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import AsyncSessionLocal, init_db
from db.models import Doctor, Slot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
MOCK_DOCTORS = [
    # Cardiology
    {"name": "Dr. Sarah Chen", "specialty": "Cardiology", "department": "Cardiology Unit"},
    {"name": "Dr. James Patel", "specialty": "Cardiology", "department": "Cardiology Unit"},
    # Emergency Medicine
    {"name": "Dr. Maria Rodriguez", "specialty": "Emergency Medicine", "department": "Emergency Department"},
    {"name": "Dr. Kevin O'Brien", "specialty": "Emergency Medicine", "department": "Emergency Department"},
    # General Practice
    {"name": "Dr. Anna Thompson", "specialty": "General Practice", "department": "Outpatient Clinic"},
    {"name": "Dr. David Kim", "specialty": "General Practice", "department": "Outpatient Clinic"},
    {"name": "Dr. Priya Sharma", "specialty": "General Practice", "department": "Outpatient Clinic"},
    # Neurology
    {"name": "Dr. Robert Walsh", "specialty": "Neurology", "department": "Neurology Department"},
    {"name": "Dr. Lisa Park", "specialty": "Neurology", "department": "Neurology Department"},
    # Pulmonology
    {"name": "Dr. Ahmed Hassan", "specialty": "Pulmonology", "department": "Respiratory Medicine"},
    # Gastroenterology
    {"name": "Dr. Emma Wilson", "specialty": "Gastroenterology", "department": "Digestive Health"},
    # Orthopaedics
    {"name": "Dr. Michael Brown", "specialty": "Orthopaedics", "department": "Musculoskeletal Unit"},
    # Dermatology
    {"name": "Dr. Yuki Tanaka", "specialty": "Dermatology", "department": "Skin & Allergy Clinic"},
    # Psychiatry
    {"name": "Dr. Sophie Laurent", "specialty": "Psychiatry", "department": "Mental Health Services"},
    # Paediatrics
    {"name": "Dr. Carlos Mendoza", "specialty": "Paediatrics", "department": "Children's Health"},
]

# Slot hours per day (9 AM to 5 PM, every 30 minutes)
SLOT_HOURS = [
    (9, 0), (9, 30),
    (10, 0), (10, 30),
    (11, 0), (11, 30),
    (13, 0), (13, 30),
    (14, 0), (14, 30),
    (15, 0), (15, 30),
    (16, 0), (16, 30),
]


def generate_slots_for_doctor(doctor_id: uuid.UUID, days_ahead: int = 7) -> list[Slot]:
    """Generate slots for the next N business days."""
    slots = []
    now = datetime.now(timezone.utc)
    days_generated = 0
    day_offset = 1

    while days_generated < days_ahead:
        candidate = now + timedelta(days=day_offset)
        day_offset += 1

        # Skip weekends
        if candidate.weekday() >= 5:  # 5=Saturday, 6=Sunday
            continue

        for hour, minute in SLOT_HOURS:
            slot_dt = candidate.replace(hour=hour, minute=minute, second=0, microsecond=0)
            slots.append(Slot(doctor_id=doctor_id, datetime=slot_dt, is_booked=False))

        days_generated += 1

    return slots


async def seed() -> None:
    """Run the full seed operation."""
    logger.info("Initialising DB tables...")
    await init_db()

    async with AsyncSessionLocal() as session:
        # Check if already seeded
        from sqlalchemy import select, func
        count_result = await session.execute(select(func.count()).select_from(Doctor))
        doctor_count = count_result.scalar()

        if doctor_count and doctor_count > 0:
            logger.info("Database already seeded (%d doctors found). Skipping.", doctor_count)
            return

        logger.info("Seeding %d doctors...", len(MOCK_DOCTORS))
        doctors = []
        for d in MOCK_DOCTORS:
            doctor = Doctor(**d)
            session.add(doctor)
            doctors.append(doctor)

        await session.flush()  # get IDs before creating slots

        slot_count = 0
        for doctor in doctors:
            slots = generate_slots_for_doctor(doctor.id)
            for slot in slots:
                session.add(slot)
            slot_count += len(slots)

        await session.commit()
        logger.info(
            "Seed complete. Created %d doctors and %d appointment slots.",
            len(doctors),
            slot_count,
        )


if __name__ == "__main__":
    asyncio.run(seed())
