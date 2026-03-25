"""
api/schema.py
──────────────
Pydantic models for request validation and response serialisation.
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator


# ── Request ───────────────────────────────────────────────────────────────

class FlowFeatures(BaseModel):
    """
    Single network flow feature vector.
    All 47 CICIoT2023 features are optional at the schema level;
    the predictor will raise a 422 if required selected features are missing.
    Field names match the column names in the parquet exactly.
    """
    flow_duration:      Optional[float] = Field(None, description="Duration of the flow in microseconds")
    Header_Length:      Optional[float] = None
    Protocol_Type:      Optional[float] = Field(None, alias="Protocol Type")
    Duration:           Optional[float] = None
    Rate:               Optional[float] = None
    Srate:              Optional[float] = None
    Drate:              Optional[float] = None
    fin_flag_number:    Optional[float] = None
    syn_flag_number:    Optional[float] = None
    rst_flag_number:    Optional[float] = None
    psh_flag_number:    Optional[float] = None
    ack_flag_number:    Optional[float] = None
    ece_flag_number:    Optional[float] = None
    cwr_flag_number:    Optional[float] = None
    ack_count:          Optional[float] = None
    syn_count:          Optional[float] = None
    fin_count:          Optional[float] = None
    urg_count:          Optional[float] = None
    rst_count:          Optional[float] = None
    HTTP:               Optional[float] = None
    HTTPS:              Optional[float] = None
    DNS:                Optional[float] = None
    Telnet:             Optional[float] = None
    SMTP:               Optional[float] = None
    SSH:                Optional[float] = None
    IRC:                Optional[float] = None
    TCP:                Optional[float] = None
    UDP:                Optional[float] = None
    DHCP:               Optional[float] = None
    ARP:                Optional[float] = None
    ICMP:               Optional[float] = None
    IPv:                Optional[float] = None
    LLC:                Optional[float] = None
    Tot_sum:            Optional[float] = Field(None, alias="Tot sum")
    Min:                Optional[float] = None
    Max:                Optional[float] = None
    AVG:                Optional[float] = None
    Std:                Optional[float] = None
    Tot_size:           Optional[float] = Field(None, alias="Tot size")
    IAT:                Optional[float] = None
    Number:             Optional[float] = None
    Magnitue:           Optional[float] = None
    Radius:             Optional[float] = None
    Covariance:         Optional[float] = None
    Variance:           Optional[float] = None
    Weight:             Optional[float] = None

    model_config = {"populate_by_name": True}


class BatchFlowRequest(BaseModel):
    """Up to 1000 flows per batch request."""
    flows: list[FlowFeatures] = Field(..., min_length=1, max_length=1000)


# ── Response ──────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    prediction:   str              = Field(..., description="Predicted attack class")
    confidence:   float            = Field(..., description="Model confidence (0–1) for predicted class")
    probabilities: Dict[str, float] = Field(..., description="Probability per class")
    is_attack:    bool             = Field(..., description="True if prediction is not Benign")
    low_confidence: bool           = Field(..., description="True if confidence < 0.6 — treat with caution")


class SinglePredictionResponse(BaseModel):
    status:  str              = "ok"
    result:  PredictionResult


class BatchPredictionResponse(BaseModel):
    status:  str                    = "ok"
    count:   int
    results: list[PredictionResult]


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    n_features:   int
    classes:      list[str]
    thresholds:   Dict[str, float]


class ModelInfoResponse(BaseModel):
    model_type:        str
    n_features:        int
    feature_names:     list[str]
    classes:           list[str]
    thresholds:        Dict[str, float]