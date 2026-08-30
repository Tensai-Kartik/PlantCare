export interface QualityMetrics {
  width: number;
  height: number;
  blur_score: number;
  brightness: number;
  contrast: number;
  vegetation_ratio: number;
  green_ratio?: number;
  background_ratio?: number;
  estimated_leaf_count?: number;
}

export type SubjectCategory =
  | 'plant'
  | 'vehicle'
  | 'animal'
  | 'electronics'
  | 'person'
  | 'furniture'
  | 'food'
  | 'manmade'
  | 'non_plant';

export type QualityReasonCode =
  | 'TOO_DARK'
  | 'TOO_BRIGHT'
  | 'BLURRY'
  | 'LEAF_TOO_SMALL'
  | 'MULTIPLE_LEAVES'
  | 'LOW_RESOLUTION'
  | 'POOR_CONTRAST'
  | 'PARTIAL_LEAF'
  | 'OBSTRUCTION'
  | 'NON_PLANT_OBJECT'
  | 'SUITABLE_PLANT';

export type PredictionState =
  | 'known_high'
  | 'known_moderate'
  | 'plant_uncertain'
  | 'plant_unsupported_condition'
  | 'non_plant';

export interface QualityCheckResult {
  is_suitable: boolean;
  suitability_score: number;
  status: 'suitable' | 'warning' | 'rejected';
  is_plant: boolean;
  detected_subject: string;
  subject_category: SubjectCategory;
  plant_confidence: number;
  reason_code: QualityReasonCode | string;
  warnings: string[];
  has_multiple_leaves: boolean;
  leaf_focus_status: string;
  issues: string[];
  positive_indicators: string[];
  metrics: QualityMetrics;
  guidance: string;
}

export interface CandidatePrediction {
  class_id: string;
  name: string;
  plant?: string;
  probability: number;
  calibrated_probability?: number;
  probability_percent: number;
}

export interface PredictionResult {
  class_id: string;
  name: string;
  scientific_name?: string;
  plant: string;
  state: PredictionState;
  confidence: number;
  raw_confidence: number;
  calibrated_confidence: number;
  confidence_percent: number;
  confidence_level: 'High Confidence' | 'Moderate Confidence' | 'Low Confidence';
  entropy: number;
  top1_top2_margin: number;
  is_healthy: boolean;
  status_message?: string;
  top_candidates: CandidatePrediction[];
}

export interface PerformanceMetrics {
  image_validation_ms: number;
  preprocessing_ms: number;
  model_inference_ms: number;
  gradcam_ms: number;
  disease_metadata_lookup_ms: number;
  gemini_ms: number;
  total_request_ms: number;
}

export interface PredictionAudit {
  request_id: string;
  model_id: string;
  model_version: string;
  prediction_state: PredictionState | string;
  raw_confidence: number;
  calibrated_confidence: number;
  temperature_applied: number;
  entropy: number;
  top1_top2_margin: number;
  suitability_score: number;
  validator_status: string;
  reason_code: string;
  gradcam_status: string;
  gemini_status: string;
  performance_metrics: PerformanceMetrics;
}

export interface ModelComparisonEntry {
  model_id: string;
  model_name: string;
  architecture: string;
  predicted_class_id: string;
  predicted_name: string;
  confidence: number;
  confidence_percent: number;
  latency_ms: number;
}

export interface ModelDisagreementResult {
  agreement_status: 'AGREED' | 'DISAGREED';
  consensus_prediction: string;
  confidence_delta_percent: number;
  comparison: ModelComparisonEntry[];
}

export interface ModelMetadata {
  id: string;
  name: string;
  architecture: string;
  version?: string;
  dataset?: string;
  dataset_version?: string;
  class_count?: number;
  training_date?: string;
  temperature?: number;
  ece?: number;
  accuracy?: number;
  weighted_f1?: number;
  latency_ms?: number;
  is_default: boolean;
}

export interface ModelListResponse {
  models: ModelMetadata[];
  default: string;
}

export interface TreatmentDetails {
  immediate_steps: string[];
  organic_options: string[];
  conventional_options: string[];
}

export interface DiseaseInfo {
  id: string;
  name: string;
  scientific_name?: string;
  plant: string;
  is_healthy: boolean;
  description: string;
  symptoms: string[];
  causes: string[];
  severity: 'Healthy' | 'Low' | 'Moderate' | 'High' | 'Critical';
  spread?: string;
  image_url?: string;
  treatment: TreatmentDetails;
  prevention: string[];
  important_notes: string[];
}

export interface DiseaseListResponse {
  total: number;
  plants: string[];
  diseases: DiseaseInfo[];
}

export interface GeminiExplanation {
  summary: string;
  interpretation: string;
  care_recommendation: string;
  powered_by_gemini: boolean;
}

export interface NonPlantDetails {
  detected_subject: string;
  category: string;
  confidence_percent: number;
  message: string;
  suggestions: string[];
}

export interface AnalysisResponse {
  success: boolean;
  is_plant: boolean;
  prediction?: PredictionResult | null;
  model?: ModelMetadata | null;
  quality: QualityCheckResult;
  disease?: DiseaseInfo | null;
  non_plant_details?: NonPlantDetails | null;
  explanation?: GeminiExplanation | null;
  gradcam_heatmap_base64?: string | null;
  comparison?: ModelDisagreementResult | null;
  audit?: PredictionAudit | null;
  timestamp: string;
}

export interface ExampleLeaf {
  id: string;
  title: string;
  plant: string;
  condition: string;
  image_url: string;
  is_healthy: boolean;
}

export type PageRoute = 
  | 'dashboard' 
  | 'analyze' 
  | 'knowledge' 
  | 'diseases' 
  | 'treatment' 
  | 'prevention';

export type ThemeMode = 'light' | 'dark';
