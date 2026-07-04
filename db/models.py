"""
SQLAlchemy ORM models for TriageAI.

Tables:
  patients      — people who call the triage system
  calls         — individual triage call sessions
  doctors       — seeded mock hospital staff
  slots         — bookable appointment slots
  appointments  — confirmed bookings linking patient + slot + call
"""
import uuid
from datetime import datetime, date, timezone
from typing import Optional
from email.utils import format_datetime, parsedate_to_datetime

from sqlalchemy import (
    String, Integer, Boolean, Date, Text,
    ForeignKey, ARRAY, TypeDecorator,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.database import Base


class RFC2822DateTime(TypeDecorator):
    """
    SQLAlchemy type decorator that stores timezone-aware datetimes
    in the database as RFC 2822 compliant VARCHAR strings, while returning
    python datetime objects when queried.
    """
    impl = String(100)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return format_datetime(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return parsedate_to_datetime(value)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dob: Mapped[date] = mapped_column(Date, nullable=False)
    phone: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        RFC2822DateTime, default=datetime.utcnow
    )

    # Relationships
    calls: Mapped[list["Call"]] = relationship("Call", back_populates="patient")
    appointments: Mapped[list["Appointment"]] = relationship(
        "Appointment", back_populates="patient"
    )

    def __repr__(self) -> str:
        return f"<Patient name={self.name!r} phone={self.phone!r}>"


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    room_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        RFC2822DateTime, default=datetime.utcnow
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        RFC2822DateTime, nullable=True
    )
    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    urgency_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    urgency_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # ICD-10 codes identified during the call
    icd_codes: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), nullable=True)

    # Relationships
    patient: Mapped[Optional["Patient"]] = relationship("Patient", back_populates="calls")
    appointment: Mapped[Optional["Appointment"]] = relationship(
        "Appointment", back_populates="call", uselist=False
    )

    def __repr__(self) -> str:
        return f"<Call room_id={self.room_id!r} urgency={self.urgency_score}>"


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    specialty: Mapped[str] = mapped_column(String(100), nullable=False)
    department: Mapped[str] = mapped_column(String(100), nullable=False)

    # Relationships
    slots: Mapped[list["Slot"]] = relationship("Slot", back_populates="doctor")

    def __repr__(self) -> str:
        return f"<Doctor name={self.name!r} specialty={self.specialty!r}>"


class Slot(Base):
    __tablename__ = "slots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False
    )
    datetime: Mapped[datetime] = mapped_column(RFC2822DateTime, nullable=False)
    is_booked: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    doctor: Mapped["Doctor"] = relationship("Doctor", back_populates="slots")
    appointment: Mapped[Optional["Appointment"]] = relationship(
        "Appointment", back_populates="slot", uselist=False
    )

    def __repr__(self) -> str:
        return f"<Slot doctor_id={self.doctor_id} datetime={self.datetime} booked={self.is_booked}>"


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False
    )
    slot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("slots.id"), nullable=False
    )
    call_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        RFC2822DateTime, default=datetime.utcnow
    )

    # Relationships
    patient: Mapped["Patient"] = relationship("Patient", back_populates="appointments")
    slot: Mapped["Slot"] = relationship("Slot", back_populates="appointment")
    call: Mapped[Optional["Call"]] = relationship("Call", back_populates="appointment")

    @property
    def doctor(self) -> Doctor:
        return self.slot.doctor

    def __repr__(self) -> str:
        return f"<Appointment patient_id={self.patient_id} slot_id={self.slot_id}>"
