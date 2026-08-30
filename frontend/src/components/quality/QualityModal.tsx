import React from 'react';
import { 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  ArrowRight, 
  RotateCcw, 
  Sun, 
  Focus, 
  Maximize, 
  Layers,
  Car,
  Dog,
  Laptop,
  User,
  Building2,
  Utensils,
  Leaf,
  ShieldAlert,
  Copy,
  Scan
} from 'lucide-react';
import { QualityCheckResult, SubjectCategory } from '../../types';

interface QualityModalProps {
  imagePreviewUrl: string;
  qualityResult: QualityCheckResult;
  onProceed: () => void;
  onRetry: () => void;
}

export const QualityModal: React.FC<QualityModalProps> = ({
  imagePreviewUrl,
  qualityResult,
  onProceed,
  onRetry
}) => {
  const isPlant = qualityResult.is_plant !== false;
  const isGood = isPlant && qualityResult.status === 'suitable';
  const isWarning = isPlant && qualityResult.status === 'warning';
  const isRejected = !isPlant || qualityResult.status === 'rejected';

  const getSubjectIcon = (category: SubjectCategory) => {
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

  const getReasonCodeBadge = (code: string) => {
    switch (code) {
      case 'TOO_DARK':
        return { label: 'Low Lighting', color: 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' };
      case 'TOO_BRIGHT':
        return { label: 'Overexposed', color: 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' };
      case 'BLURRY':
        return { label: 'Motion Blur', color: 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' };
      case 'LEAF_TOO_SMALL':
        return { label: 'Leaf Too Small', color: 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' };
      case 'MULTIPLE_LEAVES':
        return { label: 'Multiple Leaves Detected', color: 'bg-blue-100 dark:bg-blue-950/60 text-blue-800 dark:text-blue-300' };
      case 'PARTIAL_LEAF':
        return { label: 'Partial Leaf (Cut-Off)', color: 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' };
      case 'OBSTRUCTION':
        return { label: 'Obstruction / Clutter', color: 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' };
      case 'NON_PLANT_OBJECT':
        return { label: 'Non-Plant Object', color: 'bg-red-100 dark:bg-red-950/60 text-red-800 dark:text-red-300' };
      default:
        return { label: 'Plant Leaf Suitable', color: 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300' };
    }
  };

  const reasonBadge = getReasonCodeBadge(qualityResult.reason_code);

  return (
    <div className="w-full max-w-2xl mx-auto bg-surface border border-subtle rounded-3xl p-6 sm:p-8 custom-shadow animate-in fade-in zoom-in-95 duration-200">
      {/* Header Badge */}
      <div className="flex items-center justify-between pb-4 mb-6 border-b border-subtle">
        <div className="flex items-center gap-2.5">
          <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
            !isPlant 
              ? 'bg-red-100 dark:bg-red-950/60 text-red-700 dark:text-red-300'
              : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300'
          }`}>
            Multi-Signal Botanical Validation
          </span>
          <h2 className="text-base sm:text-lg font-bold text-primary-color">
            {!isPlant ? 'Subject Verification Alert' : 'Image Suitability Assessment'}
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs font-bold px-2.5 py-1 rounded-lg ${reasonBadge.color}`}>
            {reasonBadge.label}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-center">
        {/* Left: Image Thumbnail */}
        <div className="relative rounded-2xl overflow-hidden border border-subtle bg-surface-subtle shadow-md">
          <img
            src={imagePreviewUrl}
            alt="Uploaded leaf for quality evaluation"
            className="w-full h-56 object-cover"
          />
          
          {/* Badge over image */}
          <div className="absolute top-2 left-2 px-2.5 py-1 rounded-lg backdrop-blur-md text-[10px] font-bold text-white shadow-xs flex items-center gap-1 bg-black/70">
            {!isPlant ? (
              <span className="text-red-400">⚠️ Non-Plant Detected</span>
            ) : (
              <span className="text-emerald-400">🌿 Plant Specimen</span>
            )}
          </div>

          {/* Multiple Leaves Warning Tag */}
          {qualityResult.has_multiple_leaves && (
            <div className="absolute top-2 right-2 px-2 py-1 rounded-lg bg-blue-600/90 text-white backdrop-blur-xs text-[10px] font-bold flex items-center gap-1 shadow-xs">
              <Copy className="w-3 h-3" />
              <span>Multi-Leaf ({qualityResult.metrics.estimated_leaf_count ?? 2} estimated)</span>
            </div>
          )}

          <div className="absolute bottom-2 left-2 px-2 py-1 rounded-md bg-black/60 backdrop-blur-xs text-[10px] text-white font-mono">
            {qualityResult.metrics.width} × {qualityResult.metrics.height} px
          </div>
        </div>

        {/* Right: Quality & Plant Verification Feedback */}
        <div className="space-y-4">
          {!isPlant ? (
            /* Non-Plant Alert Card */
            <div className="p-4 rounded-2xl bg-red-50/90 dark:bg-red-950/40 border border-red-200 dark:border-red-900/50 space-y-2.5 animate-in fade-in duration-200">
              <div className="flex items-center gap-2 text-red-800 dark:text-red-300 font-bold text-sm">
                {getSubjectIcon(qualityResult.subject_category)}
                <span>Non-Plant Image Detected</span>
              </div>
              
              <div className="p-2.5 rounded-xl bg-white/80 dark:bg-zinc-900/80 border border-red-200/60 dark:border-red-900/30">
                <span className="text-[10px] font-bold uppercase tracking-wider text-red-600 dark:text-red-400 block mb-0.5">
                  Detected Subject:
                </span>
                <p className="text-xs font-bold text-red-950 dark:text-red-200">
                  {qualityResult.detected_subject}
                </p>
              </div>

              <p className="text-xs text-red-800 dark:text-red-300 leading-relaxed">
                {qualityResult.guidance}
              </p>
            </div>
          ) : isGood ? (
            /* Suitable Plant Image */
            <div className="p-4 rounded-2xl bg-emerald-50 dark:bg-zinc-900 border border-emerald-200 dark:border-zinc-800 space-y-2">
              <div className="flex items-center gap-2 text-emerald-800 dark:text-emerald-300 font-bold text-sm">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0" />
                <span>Verified Plant Specimen</span>
              </div>
              <p className="text-xs text-emerald-700 dark:text-zinc-300">
                {qualityResult.guidance}
              </p>
            </div>
          ) : isWarning ? (
            /* Warning Plant Image */
            <div className="p-4 rounded-2xl bg-amber-50 dark:bg-zinc-900 border border-amber-200 dark:border-zinc-800 space-y-2">
              <div className="flex items-center gap-2 text-amber-800 dark:text-amber-300 font-bold text-sm">
                <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0" />
                <span>Image quality or framing advisory</span>
              </div>
              <p className="text-xs text-amber-700 dark:text-zinc-300">
                {qualityResult.guidance}
              </p>
            </div>
          ) : (
            /* Rejected Low Quality Plant Image */
            <div className="p-4 rounded-2xl bg-red-50 dark:bg-zinc-900 border border-red-200 dark:border-zinc-800 space-y-2">
              <div className="flex items-center gap-2 text-red-800 dark:text-red-300 font-bold text-sm">
                <XCircle className="w-5 h-5 text-red-600 dark:text-red-400 shrink-0" />
                <span>Image not suitable</span>
              </div>
              <p className="text-xs text-red-700 dark:text-zinc-300">
                {qualityResult.guidance}
              </p>
            </div>
          )}

          {/* Warnings List */}
          {qualityResult.warnings && qualityResult.warnings.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[11px] font-bold uppercase tracking-wider text-muted-color">
                Targeted Diagnostics:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {qualityResult.warnings.map((warn, idx) => {
                  const b = getReasonCodeBadge(warn);
                  return (
                    <span
                      key={idx}
                      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-bold ${b.color}`}
                    >
                      <Scan className="w-3 h-3" />
                      {b.label}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* Positive Badges */}
          {qualityResult.positive_indicators.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[11px] font-bold uppercase tracking-wider text-muted-color">
                Passed Checks:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {qualityResult.positive_indicators.map((ind, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-50 dark:bg-zinc-800 text-emerald-800 dark:text-emerald-300 border border-emerald-200/60 dark:border-zinc-700 text-[11px] font-medium"
                  >
                    <CheckCircle2 className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                    {ind}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Metrics Strip */}
      <div className="mt-6 pt-4 border-t border-subtle grid grid-cols-4 gap-2 text-center text-xs">
        <div className="p-2 rounded-xl bg-surface-elevated">
          <div className="flex items-center justify-center gap-1 text-muted-color mb-0.5">
            <Focus className="w-3.5 h-3.5" />
            <span>Sharpness</span>
          </div>
          <p className="font-bold text-primary-color">{qualityResult.metrics.blur_score}</p>
        </div>
        <div className="p-2 rounded-xl bg-surface-elevated">
          <div className="flex items-center justify-center gap-1 text-muted-color mb-0.5">
            <Sun className="w-3.5 h-3.5" />
            <span>Brightness</span>
          </div>
          <p className="font-bold text-primary-color">{qualityResult.metrics.brightness}</p>
        </div>
        <div className="p-2 rounded-xl bg-surface-elevated">
          <div className="flex items-center justify-center gap-1 text-muted-color mb-0.5">
            <Layers className="w-3.5 h-3.5" />
            <span>Contrast</span>
          </div>
          <p className="font-bold text-primary-color">{qualityResult.metrics.contrast}</p>
        </div>
        <div className="p-2 rounded-xl bg-surface-elevated">
          <div className="flex items-center justify-center gap-1 text-muted-color mb-0.5">
            <Maximize className="w-3.5 h-3.5" />
            <span>Foliage %</span>
          </div>
          <p className={`font-bold ${!isPlant ? 'text-red-500' : 'text-primary-color'}`}>
            {qualityResult.metrics.vegetation_ratio}%
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-6 flex items-center justify-end gap-3">
        <button
          type="button"
          onClick={onRetry}
          className={`flex items-center gap-1.5 px-4 py-2.5 rounded-xl border text-sm font-medium transition-all ${
            !isPlant 
              ? 'bg-emerald-600 hover:bg-emerald-700 text-white font-semibold shadow-md shadow-emerald-600/20 border-transparent'
              : 'border-subtle hover:bg-surface-elevated text-secondary-color'
          }`}
        >
          {!isPlant ? (
            <>
              <Leaf className="w-4 h-4" />
              <span>Upload a Plant Leaf</span>
            </>
          ) : (
            <>
              <RotateCcw className="w-4 h-4" />
              <span>Try Another Image</span>
            </>
          )}
        </button>

        {/* If valid plant, allow continuing */}
        {isPlant && !isRejected && (
          <button
            type="button"
            onClick={onProceed}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm shadow-md shadow-emerald-600/20 hover:scale-[1.02] active:scale-[0.98] transition-all"
          >
            <span>{isWarning ? 'Continue Anyway' : 'Continue to Analysis'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};
