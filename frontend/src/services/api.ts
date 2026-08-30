import {
  AnalysisResponse,
  QualityCheckResult,
  ModelListResponse,
  ModelDisagreementResult,
  DiseaseListResponse,
  DiseaseInfo,
  ExampleLeaf
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export async function fetchHealth(): Promise<{ status: string; calibration_enabled?: boolean; version?: string }> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    console.warn('Backend offline, running in resilient mode:', err);
    return { status: 'offline', calibration_enabled: true, version: '1.2.0' };
  }
}

export async function fetchModels(): Promise<ModelListResponse> {
  try {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error('Failed to fetch models');
    return await res.json();
  } catch (err) {
    console.warn('Using fallback models metadata:', err);
    return {
      default: 'efficientnet_b0',
      models: [
        {
          id: 'efficientnet_b0',
          name: 'EfficientNet-B0',
          architecture: 'efficientnet_b0',
          version: '1.2.0',
          dataset: 'PlantVillage + FieldAug',
          dataset_version: '2.0',
          class_count: 21,
          training_date: '2026-08-26',
          temperature: 1.15,
          ece: 0.038,
          accuracy: 90.48,
          weighted_f1: 0.8845,
          latency_ms: 13.03,
          is_default: true
        },
        {
          id: 'mobilenet_v3_small',
          name: 'MobileNetV3-Small',
          architecture: 'mobilenet_v3_small',
          version: '1.2.0',
          dataset: 'PlantVillage + FieldAug',
          dataset_version: '2.0',
          class_count: 21,
          training_date: '2026-08-26',
          temperature: 1.20,
          ece: 0.052,
          accuracy: 77.78,
          weighted_f1: 0.7135,
          latency_ms: 1.91,
          is_default: false
        }
      ]
    };
  }
}

export async function checkImageQuality(file: File): Promise<QualityCheckResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/quality-check`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Image quality check failed');
  }

  return await res.json();
}

export async function analyzePlant(
  file: File,
  modelId?: string,
  skipQualityCheck: boolean = false,
  enableModelComparison: boolean = false
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (modelId) formData.append('model_id', modelId);
  formData.append('skip_quality_check', String(skipQualityCheck));
  formData.append('enable_model_comparison', String(enableModelComparison));

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Plant analysis failed');
  }

  return await res.json();
}

export async function compareModels(file: File): Promise<ModelDisagreementResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/compare-models`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Model comparison failed');
  }

  return await res.json();
}

export async function analyzeExample(
  exampleId: string,
  modelId?: string,
  enableModelComparison: boolean = false
): Promise<AnalysisResponse> {
  const url = new URL(`${API_BASE}/analyze-example/${exampleId}`);
  if (modelId) url.searchParams.append('model_id', modelId);
  if (enableModelComparison) url.searchParams.append('enable_model_comparison', 'true');

  const res = await fetch(url.toString(), {
    method: 'POST'
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Example analysis failed');
  }

  return await res.json();
}

export async function fetchDiseases(params?: {
  plant?: string;
  severity?: string;
  q?: string;
}): Promise<DiseaseListResponse> {
  try {
    const url = new URL(`${API_BASE}/diseases`);
    if (params?.plant && params.plant !== 'all') url.searchParams.append('plant', params.plant);
    if (params?.severity && params.severity !== 'all') url.searchParams.append('severity', params.severity);
    if (params?.q) url.searchParams.append('q', params.q);

    const res = await fetch(url.toString());
    if (!res.ok) throw new Error('Failed to fetch disease knowledge base');
    return await res.json();
  } catch (err) {
    console.error('Error fetching diseases:', err);
    throw err;
  }
}

export async function fetchDiseaseDetail(id: string): Promise<DiseaseInfo> {
  const res = await fetch(`${API_BASE}/diseases/${id}`);
  if (!res.ok) throw new Error(`Disease with ID ${id} not found`);
  return await res.json();
}

export async function fetchExamples(): Promise<ExampleLeaf[]> {
  try {
    const res = await fetch(`${API_BASE}/examples`);
    if (!res.ok) throw new Error('Failed to fetch examples');
    return await res.json();
  } catch (err) {
    return [
      {
        id: 'tomato_early_blight',
        title: 'Tomato Early Blight',
        plant: 'Tomato',
        condition: 'Alternaria solani',
        image_url: '/examples/tomato_early_blight.jpg',
        is_healthy: false
      },
      {
        id: 'potato_late_blight',
        title: 'Potato Late Blight',
        plant: 'Potato',
        condition: 'Phytophthora infestans',
        image_url: '/examples/potato_late_blight.jpg',
        is_healthy: false
      },
      {
        id: 'apple_scab',
        title: 'Apple Scab',
        plant: 'Apple',
        condition: 'Venturia inaequalis',
        image_url: '/examples/apple_scab.jpg',
        is_healthy: false
      },
      {
        id: 'grape_black_rot',
        title: 'Grape Black Rot',
        plant: 'Grape',
        condition: 'Guignardia bidwellii',
        image_url: '/examples/grape_black_rot.jpg',
        is_healthy: false
      },
      {
        id: 'pepper_bell_bacterial_spot',
        title: 'Pepper Bacterial Spot',
        plant: 'Pepper',
        condition: 'Xanthomonas euvesicatoria',
        image_url: '/examples/pepper_bell_bacterial_spot.jpg',
        is_healthy: false
      },
      {
        id: 'tomato_healthy',
        title: 'Tomato Healthy Leaf',
        plant: 'Tomato',
        condition: 'Solanum lycopersicum',
        image_url: '/examples/tomato_healthy.jpg',
        is_healthy: true
      }
    ];
  }
}
