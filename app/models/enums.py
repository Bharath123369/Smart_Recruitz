from enum import Enum

class ReadinessLevel(str, Enum):
    HIGH_READY = "HIGH_READY"
    INTERVIEW_READY = "INTERVIEW_READY"
    BORDERLINE = "BORDERLINE"
    NOT_READY = "NOT_READY"

class L1Status(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    NEEDS_REVIEW = "NEEDS_REVIEW"

class TalentPoolAction(str, Enum):
    INSERT = "INSERT"
    HOLD = "HOLD"
    REJECT = "REJECT"

class AnswerQuality(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    AVERAGE = "AVERAGE"
    POOR = "POOR"

class InterviewType(str, Enum):
    L1_SCREENING = "L1_SCREENING"
    TECHNICAL = "TECHNICAL"
    CULTURAL = "CULTURAL"
