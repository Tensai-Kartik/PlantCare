import React, { useState } from 'react';
import { Scale, Sparkles, ShieldCheck } from 'lucide-react';
import { LeafDropzone } from '../components/upload/LeafDropzone';
import { QualityModal } from '../components/quality/QualityModal';
import { AnalyzingScreen } from '../components/analysis/AnalyzingScreen';
import { QualityCheckResult, AnalysisResponse } from '../types';
import { checkImageQuality, analyzePlant } from '../services/api';

interface AnalyzePlantProps {
  selectedModel: string;
  modelName: string;
  onAnalysisComplete: (result: AnalysisResponse, previewUrl: string) => void;
}

export const AnalyzePlant: React.FC<AnalyzePlantProps> = ({ 
  selectedModel, 
  modelName,
  onAnalysisComplete
}) => {
  const [currentStep, setCurrentStep] = useState<'upload' | 'quality' | 'analyzing'>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [qualityResult, setQualityResult] = useState<QualityCheckResult | null>(null);
  const [enableModelComparison, setEnableModelComparison] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelected = async (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setIsProcessing(true);
    setError(null);

    try {
      // 1. Evaluate image quality & botanical presence
      const qRes = await checkImageQuality(file);
      setQualityResult(qRes);
      setCurrentStep('quality');
    } catch (err: any) {
      console.error('Quality check error:', err);
      // Fallback: Proceed directly to analysis if quality check encounters transient network lag
      handleProceedToAnalysis(file, true);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProceedToAnalysis = async (fileToUse?: File, skipQuality: boolean = false) => {
    const file = fileToUse || selectedFile;
    if (!file) return;

    setCurrentStep('analyzing');
    setIsProcessing(true);
    setError(null);

    try {
      const res = await analyzePlant(file, selectedModel, skipQuality, enableModelComparison);
      if (previewUrl) {
        onAnalysisComplete(res, previewUrl);
      }
    } catch (err: any) {
      console.error('Analysis error:', err);
      setError(err.message || 'Failed to complete plant disease analysis.');
      setCurrentStep('upload');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setCurrentStep('upload');
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setQualityResult(null);
    setError(null);
  };

  return (
    <div className="w-full space-y-6">
      {error && (
        <div className="max-w-2xl mx-auto p-4 rounded-2xl bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 text-xs border border-red-200 dark:border-red-900/50">
          <p className="font-bold mb-1">Analysis Error</p>
          <p>{error}</p>
        </div>
      )}

      {currentStep === 'upload' && (
        <div className="max-w-2xl mx-auto space-y-6 animate-in fade-in duration-300">
          <div className="text-center space-y-1">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 text-xs font-bold mb-1">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Multi-Signal Botanical Guardrails Active</span>
            </div>
            <h2 className="text-2xl font-black text-primary-color tracking-tight">
              Analyze Plant Leaf
            </h2>
            <p className="text-xs sm:text-sm text-secondary-color">
              Upload or snap a high-resolution photo of a single crop leaf for calibrated pathology diagnosis.
            </p>
          </div>

          {/* Model Consensus Toggle */}
          <div className="p-4 rounded-2xl bg-surface border border-subtle flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-surface-elevated text-primary-color">
                <Scale className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div>
                <span className="text-xs font-bold text-primary-color block">
                  Multi-Model & AI Vision Consensus
                </span>
                <span className="text-[11px] text-muted-color">
                  Cross-checks ensemble predictions across EfficientNet-B0, MobileNetV3-Small, ResNet-18, and Gemini Vision
                </span>
              </div>
            </div>

            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={enableModelComparison}
                onChange={(e) => setEnableModelComparison(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-surface-elevated peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
            </label>
          </div>

          <div className="bg-surface border border-subtle rounded-3xl p-6 sm:p-8 custom-shadow">
            <LeafDropzone onImageSelected={handleImageSelected} isLoading={isProcessing} />
          </div>
        </div>
      )}

      {currentStep === 'quality' && previewUrl && qualityResult && (
        <QualityModal
          imagePreviewUrl={previewUrl}
          qualityResult={qualityResult}
          onProceed={() => handleProceedToAnalysis(undefined, false)}
          onRetry={handleReset}
        />
      )}

      {currentStep === 'analyzing' && previewUrl && (
        <AnalyzingScreen
          imagePreviewUrl={previewUrl}
          modelName={modelName}
        />
      )}
    </div>
  );
};
