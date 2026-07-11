"""
Mock HIS (Hospital Information System) Scheduling API.

Endpoints:
  GET  /schedule/doctors?specialty={specialty}  — list doctors by specialty
  GET  /schedule/slots?doctor_id={id}           — list open slots for a doctor
  POST /schedule/book                           — book a slot

This mock mirrors what a real HIS API (e.g. Epic, Cerner) would look like.
The data lives in Postgres, seeded by scripts/seed_db.py.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, cast, DateTime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db.database import get_db
from db.models import Doctor, Slot, Appointment, Patient, Call
from api.models import (
    DoctorResponse,
    SlotResponse,
    BookAppointmentRequest,
    AppointmentResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /schedule/doctors
# ---------------------------------------------------------------------------
@router.get("/doctors", response_model=list[DoctorResponse])
async def get_doctors(
    specialty: Optional[str] = Query(default=None, description="Filter by medical specialty"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return all doctors, optionally filtered by specialty.
    Case-insensitive substring match on specialty field.
    """
    stmt = select(Doctor)
    if specialty:
        stmt = stmt.where(Doctor.specialty.ilike(f"%{specialty}%"))
    result = await db.execute(stmt)
    return result.scalars().all()


# ---------------------------------------------------------------------------
# GET /schedule/slots
# ---------------------------------------------------------------------------
@router.get("/slots", response_model=list[SlotResponse])
async def get_slots(
    doctor_id: uuid.UUID = Query(..., description="UUID of the doctor"),
    db: AsyncSession = Depends(get_db),
):
    """Return available (unbooked) slots for a given doctor, ordered by datetime."""
    stmt = (
        select(Slot)
        .where(Slot.doctor_id == doctor_id, Slot.is_booked == False)  # noqa: E712
        .order_by(cast(Slot.datetime, DateTime))
    )
    result = await db.execute(stmt)
    slots = result.scalars().all()
    if not slots:
        raise HTTPException(status_code=404, detail="No available slots for this doctor")
    return slots


# ---------------------------------------------------------------------------
# POST /schedule/book
# ---------------------------------------------------------------------------
@router.post("/book", response_model=AppointmentResponse)
async def book_appointment(
    payload: BookAppointmentRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Book a slot for a patient.
    Atomically marks the slot as booked and creates an Appointment record.
    """
    # 1. Ensure patient and call records exist (auto-heal if DB was offline at start of call)
    from datetime import date, datetime, timezone
    patient_res = await db.execute(select(Patient).where(Patient.id == payload.patient_id))
    patient = patient_res.scalar_one_or_none()
    if not patient:
        patient = Patient(
            id=payload.patient_id,
            name="Unknown Patient",
            dob=date(1900, 1, 1),
            phone=f"+1{str(uuid.uuid4().int)[:10]}",
            email="unknown@example.com",
        )
        db.add(patient)

    if payload.call_id:
        call_res = await db.execute(select(Call).where(Call.id == payload.call_id))
        call = call_res.scalar_one_or_none()
        if not call:
            call = Call(
                id=payload.call_id,
                room_id="unknown-booking",
                started_at=datetime.now(timezone.utc),
                patient_id=payload.patient_id,
            )
            db.add(call)

    # 1. Load the slot with its doctor
    stmt = (
        select(Slot)
        .where(Slot.id == payload.slot_id)
        .options(selectinload(Slot.doctor))
    )
    result = await db.execute(stmt)
    slot = result.scalar_one_or_none()

    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    if slot.is_booked:
        raise HTTPException(status_code=409, detail="Slot is already booked")

    # 2. Mark slot as booked
    slot.is_booked = True

    # 3. Create appointment
    appointment = Appointment(
        patient_id=payload.patient_id,
        slot_id=payload.slot_id,
        call_id=payload.call_id,
    )
    db.add(appointment)
    await db.flush()  # get the appointment ID before commit

    # 4. Load the appointment with all relationships for response
    await db.refresh(appointment)
    stmt2 = (
        select(Appointment)
        .where(Appointment.id == appointment.id)
        .options(
            selectinload(Appointment.slot).selectinload(Slot.doctor),
            selectinload(Appointment.patient),
        )
    )
    result2 = await db.execute(stmt2)
    full_appointment = result2.scalar_one()

    return full_appointment


# ---------------------------------------------------------------------------
# GET /schedule/appointments/{appointment_id}
# ---------------------------------------------------------------------------
@router.get("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def get_appointment(
    appointment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Fetch a single appointment by ID."""
    stmt = (
        select(Appointment)
        .where(Appointment.id == appointment_id)
        .options(
            selectinload(Appointment.slot).selectinload(Slot.doctor),
            selectinload(Appointment.patient),
        )
    )
    result = await db.execute(stmt)
    appointment = result.scalar_one_or_none()
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return appointment
