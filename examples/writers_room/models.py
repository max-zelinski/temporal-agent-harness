"""Pydantic models crossing the Temporal converter for the writers-room example.

These are the envelopes the container agents accept and return — the harness requires every
``@agent.accepts`` handler to take one pydantic input model and return one pydantic output
model, so the pipeline's contracts live here, one model per boundary.

NB: no ``from __future__ import annotations`` — stringized annotations trip the pydantic
converter and the handler-parameter schemas a UI renders forms from (same rule as
``examples/agent_dag/models.py``).
"""

from pydantic import BaseModel, Field


class PitchRequest(BaseModel):
    """What to greenlight a writers room on: the historical subject the film is about."""

    subject: str = Field(
        description="A real historical person, place, or event, e.g. 'Ada Lovelace'."
    )


class Treatment(BaseModel):
    """The screenwriter's unit of work: a logline plus a three-act treatment."""

    logline: str = Field(description="The film in one sentence.")
    treatment: str = Field(
        description="A three-act treatment of about 150 words: setup, escalation, resolution."
    )


class Critique(BaseModel):
    """The critic's verdict on one draft — the loop's exit condition."""

    approved: bool = Field(
        description="True only when the draft needs no further revision."
    )
    notes: str = Field(
        description="Specific, actionable feedback for the next draft; one line of praise when approved."
    )


class PitchResult(BaseModel):
    """The writers room's finished concept, handed to preproduction."""

    logline: str
    draft: str = Field(description="The approved treatment text.")
    iterations: int = Field(
        description="How many draft/critique cycles the room ran before approval."
    )


class PreproductionInput(BaseModel):
    """The approved concept the preproduction team analyzes."""

    logline: str
    draft: str = Field(description="The approved treatment, in full.")


class PreproductionResult(BaseModel):
    """The two analyses the preproduction team produced in parallel."""

    box_office: str = Field(
        description="Estimated worldwide box-office range and the comparable films behind it."
    )
    casting: str = Field(
        description="Casting suggestions for the principal characters."
    )


class ScreenplayPackage(BaseModel):
    """Everything the file writer needs to assemble the final pitch document."""

    subject: str
    logline: str
    draft: str
    box_office: str
    casting: str


class FileWriteResult(BaseModel):
    """What the file writer produced: a titled pitch document on disk."""

    title: str = Field(description="The title the file writer gave the film concept.")
    path: str = Field(description="Filesystem path the pitch document was written to.")
