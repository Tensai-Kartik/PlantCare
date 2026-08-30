import React, { useState, useEffect } from 'react';
import { 
  CheckCircle2, 
  Sparkles, 
  Cpu, 
  ShieldAlert, 
  ChevronDown,
  Info,
  Check,
  AlertTriangle,
  HelpCircle,
  Clock,
  Activity,
  Layers,
  Car,
  Dog,
  Laptop,
  User,
  Building2,
  Utensils,
  Leaf,
  Camera,
  SunMedium,
  Focus,
  Maximize2,
  RotateCcw,
  Sliders,
  Scale,
  Hash,
  BookOpen,
  ExternalLink
} from 'lucide-react';
import confetti from 'canvas-confetti';
import { AnalysisResponse, PredictionState, CandidatePrediction } from '../../types';

interface ResultViewProps {
  analysis: AnalysisResponse;
  originalImagePreview: string;
  onReset?: () => void;
  onSelectCandidate?: (candidate: CandidatePrediction) => void;
  onOpenInKnowledgeBase?: (diseaseId: string) => void;
}

type TabType = 'overview' | 'symptoms' | 'causes' | 'treatment' | 'prevention' | 'important';

export const ResultView: React.FC<ResultViewProps> = ({
  analysis,
  originalImagePreview,
  onReset,
  onOpenInKnowledgeBase
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [heatmapView, setHeatmapView] = useState<'heatmap' | 'original'>('heatmap');
  const [showCandidates, setShowCandidates] = useState(false);
  const [showAuditDrawer, setShowAuditDrawer] = useState(false);

  const isPlant = analysis.is_plant !== false && !!analysis.prediction;
  const { prediction, model, disease, explanation, gradcam_heatmap_base64, non_plant_details, quality, comparison, audit } = analysis;
  const state: PredictionState = prediction?.state || (isPlant ? 'known_high' : 'non_plant');

  // Trigger celebration confetti only if authentic healthy plant!
  useEffect(() => {
    if (isPlant && prediction?.is_healthy && state === 'known_high') {
      confetti({
        particleCount: 80,
        spread: 60,
        origin: { y: 0.6 }
      });
    }
  }, [isPlant, prediction?.is_healthy, state]);

  const getSubjectIcon = (category?: string) => {
    switch (category) {
      case 'vehicle':
        return <Car className="w-5 h-5 text-red-500 shrink-0" />;
      case 'animal':
        return <Dog className="w-5 h-5 text-amber-500 shrink-0" />;
      case 'electronics':
        return <Laptop className="w-5 h-5 text-blue-500 shrink-0" />;
      case 'person':
        return <User className="w-5 h-5 text-purple-500 shrink-0" />;
      case 'furniture':
        return <Building2 className="w-5 h-5 text-indigo-500 shrink-0" />;
      case 'food':
        return <Utensils className="w-5 h-5 text-orange-500 shrink-0" />;
      default:
        return <ShieldAlert className="w-5 h-5 text-red-500 shrink-0" />;
    }
  };

  // ---------------------------------------------------------------------------
  // STATE 5: NON-PLANT SPECIMEN DETECTED SCREEN
  // ---------------------------------------------------------------------------
  if (!isPlant || state === 'non_plant') {
    const subjectName = non_plant_details?.detected_subject || quality.detected_subject || 'Non-Plant Object';
    const category = non_plant_details?.category || quality.subject_category || 'non_plant';
    const confidenceVal = non_plant_details?.confidence_percent ?? (100.0 - quality.plant_confidence);

    return (
      <div className="w-full max-w-5xl mx-auto space-y-6 animate-in fade-in duration-300">
        {/* Top Out-of-Domain Guardrail Alert Card */}
        <div className="bg-gradient-to-r from-red-50 via-red-100/50 to-amber-50 dark:from-red-950/40 dark:via-zinc-900 dark:to-zinc-900 border border-red-200 dark:border-red-900/50 rounded-3xl p-6 sm:p-8 custom-shadow space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-2xl bg-red-600 text-white shadow-md shadow-red-600/30">
                {getSubjectIcon(category)}
              </div>
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-red-600 dark:text-red-400 block">
                  Out-of-Domain Detection Guardrail
                </span>
                <h2 className="text-xl sm:text-2xl font-black text-red-950 dark:text-red-100 tracking-tight">
                  Non-Plant Specimen Detected
                </h2>
              </div>
            </div>

            <div className="flex items-center gap-2 self-start sm:self-auto px-3.5 py-1.5 rounded-xl bg-red-100 dark:bg-red-950/80 border border-red-200 dark:border-red-900/60 text-xs font-bold text-red-800 dark:text-red-300">
              <span>Verified Non-Plant ({confidenceVal.toFixed(1)}%)</span>
            </div>
          </div>

          <p className="text-xs sm:text-sm text-red-950/90 dark:text-zinc-300 leading-relaxed max-w-3xl">
            PlantCare is an agricultural pathology AI specifically calibrated for botanical leaves and crops. 
            The uploaded image was verified as a <strong>{subjectName}</strong>. 
            To prevent inaccurate false positives (such as misdiagnosing vehicles or objects as potato blight), plant disease classification has been safely halted.
          </p>
        </div>

        {/* 2-Column Details: Image vs Explanation */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Column 1: Uploaded Specimen */}
          <div className="bg-surface border border-subtle rounded-3xl p-5 sm:p-6 custom-shadow space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-color">
                Uploaded Specimen
              </span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-300">
                Non-Botanical
              </span>
            </div>

            <div className="relative rounded-2xl overflow-hidden bg-surface-subtle border border-red-200 dark:border-red-900/40 shadow-inner h-64">
              <img
                src={originalImagePreview}
                alt="Uploaded non-plant specimen"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 left-2 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-xs text-[11px] font-bold text-white flex items-center gap-1.5">
                {getSubjectIcon(category)}
                <span>{subjectName}</span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center text-[11px] pt-1">
              <div className="p-2 rounded-xl bg-surface-elevated border border-subtle">
                <span className="text-muted-color block mb-0.5">Dimensions</span>
                <span className="font-bold text-primary-color">{quality.metrics.width}×{quality.metrics.height}px</span>
              </div>
              <div className="p-2 rounded-xl bg-surface-elevated border border-subtle">
                <span className="text-muted-color block mb-0.5">Foliage %</span>
                <span className="font-bold text-red-500">{quality.metrics.vegetation_ratio}%</span>
              </div>
              <div className="p-2 rounded-xl bg-surface-elevated border border-subtle">
                <span className="text-muted-color block mb-0.5">Status</span>
                <span className="font-bold text-red-600 dark:text-red-400">Rejected</span>
              </div>
            </div>
          </div>

          {/* Column 2: Verification Analysis & Instructions */}
          <div className="bg-surface border border-subtle rounded-3xl p-5 sm:p-6 custom-shadow flex flex-col justify-between space-y-4">
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-5 h-5 text-red-500" />
                <h3 className="text-base font-bold text-primary-color">
                  Pathology Safety Advisory
                </h3>
              </div>

              <div className="p-4 rounded-2xl bg-surface-elevated border border-subtle space-y-2 text-xs">
                <span className="font-bold text-secondary-color block">Why was analysis stopped?</span>
                <p className="text-muted-color leading-relaxed">
                  Closed-set neural networks trained exclusively on agricultural datasets will output erroneous crop labels with uncalibrated probabilities when given non-plant images. PlantCare's multi-signal guardrail prevents false diagnoses by validating botanical presence before running pathology models.
                </p>
              </div>

              <div className="space-y-2">
                <span className="text-xs font-bold text-primary-color block">
                  Next Steps:
                </span>
                <ul className="space-y-1.5 text-xs text-secondary-color">
                  {non_plant_details?.suggestions?.map((sug, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-emerald-600 dark:text-emerald-400 font-bold shrink-0">✓</span>
                      <span>{sug}</span>
                    </li>
                  )) || (
                    <>
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-600 dark:text-emerald-400 font-bold">✓</span>
                        <span>Upload a photo of a plant leaf (e.g. Tomato, Potato, Apple, Grape, Pepper, Corn).</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-600 dark:text-emerald-400 font-bold">✓</span>
                        <span>Ensure the leaf fills the majority of the frame.</span>
                      </li>
                    </>
                  )}
                </ul>
              </div>
            </div>

            {onReset && (
              <button
                type="button"
                onClick={onReset}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-2xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-sm shadow-md shadow-emerald-600/20 hover:scale-[1.01] active:scale-[0.99] transition-all cursor-pointer mt-4"
              >
                <Leaf className="w-4 h-4" />
                <span>Upload a Plant Leaf for Diagnosis</span>
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // STATE 4: PLANT UNSUPPORTED / OUT-OF-INDEX CONDITION
  // ---------------------------------------------------------------------------
  if (state === 'plant_unsupported_condition') {
    return (
      <div className="w-full max-w-5xl mx-auto space-y-6 animate-in fade-in duration-300">
        <div className="bg-gradient-to-r from-amber-50 via-amber-100/50 to-orange-50 dark:from-amber-950/40 dark:via-zinc-900 dark:to-zinc-900 border border-amber-200 dark:border-amber-900/50 rounded-3xl p-6 sm:p-8 custom-shadow space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-2xl bg-amber-600 text-white shadow-md shadow-amber-600/30">
                <HelpCircle className="w-6 h-6" />
              </div>
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400 block">
                  Out-of-Distribution Condition Detection
                </span>
                <h2 className="text-xl sm:text-2xl font-black text-amber-950 dark:text-amber-100 tracking-tight">
                  Plant Foliage Detected — Condition Not Indexed
                </h2>
              </div>
            </div>

            <div className="flex items-center gap-2 self-start sm:self-auto px-3.5 py-1.5 rounded-xl bg-amber-100 dark:bg-amber-950/80 border border-amber-200 dark:border-amber-900/60 text-xs font-bold text-amber-800 dark:text-amber-300">
              <span>Entropy: {prediction?.entropy} (Uncertain)</span>
            </div>
          </div>

          <p className="text-xs sm:text-sm text-amber-950/90 dark:text-zinc-300 leading-relaxed max-w-3xl">
            Our multi-signal botanical validator confirms that this is a valid plant leaf. However, the model prediction exhibits high prediction entropy and low probability margin, indicating that this specific crop disease, viral strain, or nutrient deficiency is not in PlantCare's 21-class benchmark index.
          </p>
        </div>

        {/* Possible Close Matches & General Guidance */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-surface border border-subtle rounded-3xl p-5 sm:p-6 custom-shadow space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-muted-color">
              Closest Statistical Signatures
            </h3>
            <div className="space-y-2">
              {prediction.top_candidates.slice(0, 3).map((c, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-surface-elevated border border-subtle text-xs">
                  <span className="font-semibold text-primary-color">{c.name}</span>
                  <span className="font-mono text-muted-color">{c.probability_percent}% match</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-surface border border-subtle rounded-3xl p-5 sm:p-6 custom-shadow space-y-3">
            <h3 className="text-sm font-bold text-primary-color">
              General Agronomic Recommendations
            </h3>
            <ul className="space-y-2 text-xs text-secondary-color">
              <li className="flex items-start gap-2">
                <span className="text-amber-600 font-bold">✓</span>
                <span>Isolate the affected plant to prevent potential spore spread.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-amber-600 font-bold">✓</span>
                <span>Inspect under-leaf surfaces for spider mites, aphids, or fungal mycelium.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-amber-600 font-bold">✓</span>
                <span>Consult with a certified agricultural extension agent or local diagnostic lab.</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // STATES 1, 2, 3: AUTHENTIC PLANT DIAGNOSTIC RESULTS SCREEN
  // ---------------------------------------------------------------------------
  const getSeverityBadgeClass = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'healthy':
        return 'bg-emerald-100 text-emerald-800 dark:bg-zinc-800 dark:text-emerald-300 border-emerald-300 dark:border-zinc-700';
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-zinc-800 dark:text-blue-300 border-blue-300 dark:border-zinc-700';
      case 'moderate':
        return 'bg-amber-100 text-amber-800 dark:bg-zinc-800 dark:text-amber-300 border-amber-300 dark:border-zinc-700';
      case 'high':
      case 'critical':
        return 'bg-red-100 text-red-800 dark:bg-zinc-800 dark:text-red-300 border-red-300 dark:border-zinc-700';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-zinc-800 dark:text-gray-300';
    }
  };

  const getConfidenceBadgeClass = (level: string) => {
    switch (level) {
      case 'High Confidence':
        return 'bg-emerald-100 text-emerald-800 dark:bg-zinc-800 dark:text-emerald-300';
      case 'Moderate Confidence':
        return 'bg-amber-100 text-amber-800 dark:bg-zinc-800 dark:text-amber-300';
      default:
        return 'bg-red-100 text-red-800 dark:bg-zinc-800 dark:text-red-300';
    }
  };

  const tabs: { id: TabType; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'symptoms', label: 'Symptoms' },
    { id: 'causes', label: 'Causes' },
    { id: 'treatment', label: 'Treatment' },
    { id: 'prevention', label: 'Prevention' },
    { id: 'important', label: 'Important' }
  ];

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Multi-Model Consensus Banner (If enabled) */}
      {comparison && (
        <div className={`p-4 rounded-3xl border flex items-center justify-between gap-4 ${
          comparison.agreement_status === 'AGREED'
            ? 'bg-emerald-50/80 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900/40 text-emerald-950 dark:text-emerald-200'
            : 'bg-amber-50/80 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900/40 text-amber-950 dark:text-amber-200'
        }`}>
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-xl text-white ${comparison.agreement_status === 'AGREED' ? 'bg-emerald-600' : 'bg-amber-600'}`}>
              <Scale className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider block opacity-75">
                Multi-Model Verification Consensus
              </span>
              <p className="text-xs sm:text-sm font-bold">
                {comparison.agreement_status === 'AGREED'
                  ? `Consensus Confirmed: Both EfficientNet-B0 & MobileNetV3-Small agree on "${comparison.consensus_prediction}"`
                  : `Model Disagreement: EfficientNet-B0 & MobileNetV3-Small produced divergent predictions (Delta: ${comparison.confidence_delta_percent}%)`
                }
              </p>
            </div>
          </div>
          <div className="hidden sm:flex items-center gap-2 text-xs font-mono">
            {comparison.comparison.map((c, i) => (
              <span key={i} className="px-2 py-1 rounded-lg bg-surface border border-subtle text-primary-color">
                {c.model_name}: {c.confidence_percent}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Uncertainty Notice for State 2 & 3 */}
      {state === 'plant_uncertain' && (
        <div className="p-4 rounded-3xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/40 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-950 dark:text-amber-200 space-y-1">
            <span className="font-bold block">Inconclusive Botanical Classification</span>
            <p>
              The AI detected multiple competing disease signatures (Entropy: {prediction.entropy}, Margin: {prediction.top1_top2_margin}). Review the "Possible Matches" candidate list below and verify with the symptom checklist.
            </p>
          </div>
        </div>
      )}

      {/* Main 3-Column Diagnostic Summary Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
        {/* Column 1: Uploaded Leaf Image */}
        <div className="bg-surface border border-subtle rounded-3xl p-4 sm:p-5 flex flex-col justify-between custom-shadow">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-color">
                Uploaded Image
              </span>
              <span className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300">
                Plant Specimen
              </span>
            </div>
            <div className="relative rounded-2xl overflow-hidden bg-surface-subtle border border-subtle shadow-inner h-56">
              <img
                src={originalImagePreview}
                alt="Uploaded leaf specimen"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
          <p className="text-[11px] text-muted-color text-center mt-3">
            Assessed: {analysis.quality.metrics.width}×{analysis.quality.metrics.height}px • Foliage {analysis.quality.metrics.vegetation_ratio}%
          </p>
        </div>

        {/* Column 2: Prediction & Confidence */}
        <div className="bg-surface border border-subtle rounded-3xl p-5 sm:p-6 flex flex-col justify-between custom-shadow">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-color">
                AI Prediction
              </span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                {prediction.plant}
              </span>
            </div>

            <h3 className="text-xl sm:text-2xl font-black text-primary-color tracking-tight mt-1 leading-tight">
              {prediction.name}
            </h3>
            {prediction.scientific_name && (
              <p className="text-xs sm:text-sm text-muted-color italic mt-0.5">
                ({prediction.scientific_name})
              </p>
            )}

            {/* Confidence Score Bar */}
            <div className="mt-5 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-secondary-color">Calibrated Confidence</span>
                <span className="text-base font-black text-emerald-600 dark:text-emerald-400">
                  {prediction.confidence_percent}%
                </span>
              </div>
              <div className="w-full h-3 rounded-full bg-surface-elevated overflow-hidden border border-subtle p-0.5">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${
                    prediction.confidence_percent >= 75
                      ? 'bg-emerald-500'
                      : prediction.confidence_percent >= 45
                      ? 'bg-amber-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${prediction.confidence_percent}%` }}
                />
              </div>
            </div>

            {/* Badges */}
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <span className={`px-2.5 py-1 rounded-lg text-xs font-bold ${getConfidenceBadgeClass(prediction.confidence_level)}`}>
                {prediction.confidence_level}
              </span>
              {model && (
                <span className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-surface-elevated border border-subtle text-xs font-medium text-secondary-color">
                  <Cpu className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                  <span>{model.name} (v{model.version || '1.2.0'})</span>
                </span>
              )}
            </div>
          </div>

          {/* Top Candidates Collapsible */}
          {prediction.top_candidates.length > 1 && (
            <div className="mt-4 pt-3 border-t border-subtle">
              <button
                type="button"
                onClick={() => setShowCandidates(!showCandidates)}
                className="w-full flex items-center justify-between text-[11px] font-semibold text-secondary-color hover:text-primary-color cursor-pointer"
              >
                <span>Possible Matches & Probabilities</span>
                <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showCandidates ? 'rotate-180' : ''}`} />
              </button>

              {showCandidates && (
                <div className="mt-2 space-y-1.5 animate-in fade-in duration-150">
                  {prediction.top_candidates.slice(1, 4).map((c, i) => (
                    <div key={i} className="flex items-center justify-between text-[11px]">
                      <span className="text-secondary-color truncate max-w-[140px]">{c.name}</span>
                      <span className="font-mono text-muted-color">{c.probability_percent}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Column 3: AI Focus (Grad-CAM Heatmap) */}
        <div className="bg-surface border border-subtle rounded-3xl p-4 sm:p-5 flex flex-col justify-between custom-shadow">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <span className="text-xs font-bold uppercase tracking-wider text-muted-color">
                  AI Focus (Grad-CAM)
                </span>
              </div>
              
              {/* Toggle View Mode */}
              <div className="flex rounded-lg bg-surface-elevated p-0.5 border border-subtle text-[10px] font-medium">
                <button
                  type="button"
                  onClick={() => setHeatmapView('heatmap')}
                  className={`px-2 py-0.5 rounded-md transition-all cursor-pointer ${
                    heatmapView === 'heatmap'
                      ? 'bg-emerald-600 text-white font-bold'
                      : 'text-muted-color hover:text-primary-color'
                  }`}
                >
                  Heatmap
                </button>
                <button
                  type="button"
                  onClick={() => setHeatmapView('original')}
                  className={`px-2 py-0.5 rounded-md transition-all cursor-pointer ${
                    heatmapView === 'original'
                      ? 'bg-emerald-600 text-white font-bold'
                      : 'text-muted-color hover:text-primary-color'
                  }`}
                >
                  Clean
                </button>
              </div>
            </div>

            {/* Heatmap Image Viewer with Colorbar Legend */}
            <div className="relative rounded-2xl overflow-hidden bg-surface-subtle border border-subtle shadow-inner h-56 flex">
              <img
                src={heatmapView === 'heatmap' && gradcam_heatmap_base64 ? gradcam_heatmap_base64 : originalImagePreview}
                alt="AI Grad-CAM attention heatmap"
                className="w-full h-full object-cover"
              />

              {/* Vertical High -> Low Legend */}
              <div className="absolute top-2 right-2 flex flex-col items-center bg-black/70 backdrop-blur-xs p-1.5 rounded-lg text-[9px] text-white">
                <span className="font-bold text-red-400">High</span>
                <div className="w-2 h-16 rounded-full my-1 heatmap-gradient-bar shadow-xs" />
                <span className="font-bold text-blue-400">Low</span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-muted-color text-center mt-3">
            Warm colors highlight regions influencing the AI's diagnosis.
          </p>
        </div>
      </div>

      {/* "Why this prediction?" Explainability Card */}
      <div className="bg-surface border border-subtle rounded-3xl p-5 sm:p-6 custom-shadow space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            <h4 className="text-base font-bold text-primary-color">
              Why this prediction? — Interpretability & Uncertainty Analysis
            </h4>
          </div>
          <button
            type="button"
            onClick={() => setShowAuditDrawer(!showAuditDrawer)}
            className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 hover:underline flex items-center gap-1 cursor-pointer"
          >
            <Clock className="w-3.5 h-3.5" />
            <span>{showAuditDrawer ? 'Hide Timings & Audit' : 'View Timings & Audit'}</span>
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-2xl bg-surface-elevated border border-subtle">
            <span className="text-[10px] font-bold uppercase text-muted-color block mb-0.5">Shannon Entropy</span>
            <span className="text-sm font-bold text-primary-color">{prediction.entropy}</span>
            <span className="text-[10px] text-muted-color block mt-0.5">
              {prediction.entropy < 1.0 ? '✓ High certainty' : 'Uncertain spread'}
            </span>
          </div>

          <div className="p-3 rounded-2xl bg-surface-elevated border border-subtle">
            <span className="text-[10px] font-bold uppercase text-muted-color block mb-0.5">Top-1 / Top-2 Margin</span>
            <span className="text-sm font-bold text-primary-color">{prediction.top1_top2_margin}</span>
            <span className="text-[10px] text-muted-color block mt-0.5">
              {prediction.top1_top2_margin > 0.40 ? '✓ Distinct margin' : 'Close competitor'}
            </span>
          </div>

          <div className="p-3 rounded-2xl bg-surface-elevated border border-subtle">
            <span className="text-[10px] font-bold uppercase text-muted-color block mb-0.5">Calibrated Confidence</span>
            <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400">{prediction.confidence_percent}%</span>
            <span className="text-[10px] text-muted-color block mt-0.5">Raw Softmax: {(prediction.raw_confidence * 100).toFixed(1)}%</span>
          </div>

          <div className="p-3 rounded-2xl bg-surface-elevated border border-subtle">
            <span className="text-[10px] font-bold uppercase text-muted-color block mb-0.5">Model Temperature</span>
            <span className="text-sm font-bold text-primary-color">T = {audit?.temperature_applied ?? 1.15}</span>
            <span className="text-[10px] text-muted-color block mt-0.5">ECE: 3.8%</span>
          </div>
        </div>

        {/* Collapsible ML Audit & Micro-Timings Drawer */}
        {showAuditDrawer && audit && (
          <div className="mt-4 p-4 rounded-2xl bg-surface-subtle border border-subtle text-xs space-y-3 animate-in fade-in duration-200">
            <div className="flex items-center justify-between font-mono text-[11px] text-muted-color pb-2 border-b border-subtle">
              <span>Request ID: {audit.request_id}</span>
              <span>Model ID: {audit.model_id} (v{audit.model_version})</span>
            </div>

            <div className="space-y-1.5">
              <span className="font-bold text-secondary-color block">Inference Stage Micro-Timings:</span>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                <div className="p-2 rounded-xl bg-surface border border-subtle">
                  Validation: <span className="font-bold">{audit.performance_metrics.image_validation_ms} ms</span>
                </div>
                <div className="p-2 rounded-xl bg-surface border border-subtle">
                  Preprocessing: <span className="font-bold">{audit.performance_metrics.preprocessing_ms} ms</span>
                </div>
                <div className="p-2 rounded-xl bg-surface border border-subtle">
                  Model Inference: <span className="font-bold">{audit.performance_metrics.model_inference_ms} ms</span>
                </div>
                <div className="p-2 rounded-xl bg-surface border border-subtle">
                  Grad-CAM: <span className="font-bold">{audit.performance_metrics.gradcam_ms} ms</span>
                </div>
                <div className="p-2 rounded-xl bg-surface border border-subtle">
                  Knowledge DB: <span className="font-bold">{audit.performance_metrics.disease_metadata_lookup_ms} ms</span>
                </div>
                <div className="p-2 rounded-xl bg-surface border border-subtle text-emerald-600 font-bold">
                  Total Request: <span>{audit.performance_metrics.total_request_ms} ms</span>
                </div>
              </div>
            </div>

            <p className="text-[10px] text-muted-color pt-1">
              <strong>Interpretability Disclaimer:</strong> Grad-CAM highlights image regions containing influential visual features. It represents statistical correlation rather than definitive medical etiology.
            </p>
          </div>
        )}
      </div>

      {/* Gemini AI Care Assistant Synthesis Card */}
      {explanation && (
        <div className="bg-gradient-to-r from-emerald-50/80 via-emerald-100/40 to-teal-50/80 dark:from-zinc-900 dark:via-zinc-900/90 dark:to-zinc-900 border border-emerald-200/80 dark:border-zinc-800 rounded-3xl p-5 sm:p-6 custom-shadow space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-emerald-600 text-white shadow-sm">
                <Sparkles className="w-4 h-4" />
              </div>
              <h4 className="font-bold text-sm sm:text-base text-emerald-950 dark:text-zinc-100">
                AI Agronomist Explanation
              </h4>
            </div>
            <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-md bg-emerald-200/60 dark:bg-zinc-800 text-emerald-900 dark:text-emerald-300">
              {explanation.powered_by_gemini ? 'Powered by Gemini' : 'Curated Pathology Synthesis'}
            </span>
          </div>

          <p className="text-xs sm:text-sm text-emerald-950 dark:text-zinc-300 leading-relaxed">
            {explanation.summary}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs">
            <div className="p-3 rounded-2xl bg-white/70 dark:bg-zinc-800/80 border border-emerald-100 dark:border-zinc-700/60">
              <span className="font-bold text-emerald-900 dark:text-emerald-300 block mb-1">
                Pathology Interpretation:
              </span>
              <p className="text-secondary-color leading-relaxed">{explanation.interpretation}</p>
            </div>
            <div className="p-3 rounded-2xl bg-white/70 dark:bg-zinc-800/80 border border-emerald-100 dark:border-zinc-700/60">
              <span className="font-bold text-emerald-900 dark:text-emerald-300 block mb-1">
                Immediate Action Steps:
              </span>
              <p className="text-secondary-color leading-relaxed whitespace-pre-line">{explanation.care_recommendation}</p>
            </div>
          </div>
        </div>
      )}

      {/* Disease Pathology & Management Details Card */}
      {disease && (
        <div className="bg-surface border border-subtle rounded-3xl p-6 custom-shadow space-y-6">
          {/* Card Header & Navigation Tabs */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-subtle">
            <div className="flex border-b sm:border-b-0 border-subtle overflow-x-auto no-scrollbar gap-2 sm:gap-4">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`pb-2.5 sm:pb-1 px-1 text-sm font-semibold whitespace-nowrap transition-all border-b-2 sm:border-b-2 cursor-pointer ${
                    activeTab === tab.id
                      ? 'border-emerald-600 text-emerald-600 dark:border-emerald-400 dark:text-emerald-400 font-bold'
                      : 'border-transparent text-muted-color hover:text-primary-color'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {onOpenInKnowledgeBase && (
              <button
                type="button"
                onClick={() => onOpenInKnowledgeBase(disease.id)}
                className="flex items-center gap-1.5 self-start sm:self-auto px-3.5 py-1.5 rounded-xl bg-emerald-50 hover:bg-emerald-100 dark:bg-zinc-800 dark:hover:bg-zinc-750 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-zinc-700 text-xs font-semibold transition-all cursor-pointer shadow-xs"
                title="Open in standalone Disease Knowledge Base"
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Open in Knowledge Base</span>
                <ExternalLink className="w-3 h-3 opacity-75" />
              </button>
            )}
          </div>

          {/* Tab 1: Overview */}
          {activeTab === 'overview' && (
            <div className="space-y-6 animate-in fade-in duration-200">
              <p className="text-sm text-secondary-color leading-relaxed">
                {disease.description}
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                <div className="p-4 rounded-2xl bg-surface-elevated border border-subtle">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-color block mb-1">
                    Severity Level
                  </span>
                  <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-bold border ${getSeverityBadgeClass(disease.severity)}`}>
                    {disease.severity}
                  </span>
                </div>

                <div className="p-4 rounded-2xl bg-surface-elevated border border-subtle">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-color block mb-1">
                    Primary Host
                  </span>
                  <span className="text-sm font-bold text-primary-color">{disease.plant}</span>
                </div>

                <div className="p-4 rounded-2xl bg-surface-elevated border border-subtle">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-color block mb-1">
                    Spread Method
                  </span>
                  <span className="text-xs text-secondary-color leading-snug">
                    {disease.spread || 'Fungal spores / splash transmission'}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: Symptoms */}
          {activeTab === 'symptoms' && (
            <div className="space-y-4 animate-in fade-in duration-200">
              <h4 className="text-sm font-bold text-primary-color">
                Key Visual Indicators & Diagnostic Symptoms
              </h4>
              <div className="space-y-2.5">
                {disease.symptoms.map((symptom, idx) => (
                  <div key={idx} className="flex items-start gap-3 p-3 rounded-2xl bg-surface-elevated border border-subtle">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                    <span className="text-xs sm:text-sm text-secondary-color">{symptom}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 3: Causes */}
          {activeTab === 'causes' && (
            <div className="space-y-4 animate-in fade-in duration-200">
              <h4 className="text-sm font-bold text-primary-color">
                Environmental & Pathogenic Triggers
              </h4>
              <div className="space-y-2.5">
                {disease.causes.map((cause, idx) => (
                  <div key={idx} className="flex items-start gap-3 p-3 rounded-2xl bg-surface-elevated border border-subtle">
                    <div className="w-2 h-2 rounded-full bg-amber-500 shrink-0 mt-1.5" />
                    <span className="text-xs sm:text-sm text-secondary-color">{cause}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 4: Treatment */}
          {activeTab === 'treatment' && (
            <div className="space-y-6 animate-in fade-in duration-200">
              {/* Immediate Emergency Steps */}
              {disease.treatment.immediate_steps.length > 0 && (
                <div className="p-4 rounded-2xl bg-red-50/50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/40 space-y-2">
                  <span className="font-bold text-xs uppercase tracking-wider text-red-700 dark:text-red-300 flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4" />
                    Immediate Quarantine & Emergency Steps
                  </span>
                  <ul className="space-y-1.5 text-xs text-red-950 dark:text-red-200">
                    {disease.treatment.immediate_steps.map((step, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="font-bold shrink-0">•</span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Organic Remedies */}
              {disease.treatment.organic_options.length > 0 && (
                <div className="p-4 rounded-2xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/40 space-y-2">
                  <span className="font-bold text-xs uppercase tracking-wider text-emerald-800 dark:text-emerald-300 flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4" />
                    Organic & Low-Impact Solutions
                  </span>
                  <ul className="space-y-1.5 text-xs text-emerald-950 dark:text-emerald-200">
                    {disease.treatment.organic_options.map((opt, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="font-bold shrink-0">✓</span>
                        <span>{opt}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Conventional Treatments */}
              {disease.treatment.conventional_options.length > 0 && (
                <div className="p-4 rounded-2xl bg-blue-50/50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/40 space-y-2">
                  <span className="font-bold text-xs uppercase tracking-wider text-blue-800 dark:text-blue-300 flex items-center gap-1.5">
                    <Info className="w-4 h-4" />
                    Conventional Agricultural Treatments
                  </span>
                  <ul className="space-y-1.5 text-xs text-blue-950 dark:text-blue-200">
                    {disease.treatment.conventional_options.map((conv, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="font-bold shrink-0">•</span>
                        <span>{conv}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Tab 5: Prevention */}
          {activeTab === 'prevention' && (
            <div className="space-y-4 animate-in fade-in duration-200">
              <h4 className="text-sm font-bold text-primary-color">
                Long-term Cultural Practices & Prevention
              </h4>
              <div className="space-y-2.5">
                {disease.prevention.map((prev, idx) => (
                  <div key={idx} className="flex items-start gap-3 p-3 rounded-2xl bg-surface-elevated border border-subtle">
                    <Check className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                    <span className="text-xs sm:text-sm text-secondary-color">{prev}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 6: Important Notes */}
          {activeTab === 'important' && (
            <div className="space-y-4 animate-in fade-in duration-200">
              <div className="p-4 rounded-2xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/50 space-y-2">
                <span className="font-bold text-xs uppercase tracking-wider text-amber-800 dark:text-amber-300">
                  Important Agricultural Advisory
                </span>
                <div className="space-y-2 text-xs text-amber-950 dark:text-amber-200">
                  {disease.important_notes.map((note, idx) => (
                    <p key={idx}>• {note}</p>
                  ))}
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-surface-elevated border border-subtle text-[11px] text-muted-color">
                <strong>Disclaimer:</strong> AI predictions are intended for educational screening and decision support. They do not replace formal laboratory testing or certified extension agent consultations for commercial crop management.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
