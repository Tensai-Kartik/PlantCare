import React, { useState, useEffect } from 'react';
import { CheckCircle2, Loader2, Sparkles } from 'lucide-react';

interface AnalyzingScreenProps {
  imagePreviewUrl: string;
  modelName: string;
}

export const AnalyzingScreen: React.FC<AnalyzingScreenProps> = ({ imagePreviewUrl, modelName }) => {
  const [stage, setStage] = useState(0);

  const stages = [
    { label: 'Image received', desc: 'Decoding image buffer & verifying integrity' },
    { label: 'Preprocessing image', desc: 'Resizing to 224x224 & normalizing color channels' },
    { label: 'Extracting features', desc: 'Passing through convolutional neural network layers' },
    { label: 'Running AI model', desc: `Inference on ${modelName} & calculating probabilities` },
    { label: 'Generating results', desc: 'Computing Grad-CAM attention heatmap & pathology synthesis' }
  ];

  useEffect(() => {
    const timer1 = setTimeout(() => setStage(1), 400);
    const timer2 = setTimeout(() => setStage(2), 900);
    const timer3 = setTimeout(() => setStage(3), 1500);
    const timer4 = setTimeout(() => setStage(4), 2200);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, []);

  return (
    <div className="w-full max-w-lg mx-auto bg-surface border border-subtle rounded-3xl p-8 custom-shadow text-center space-y-6 animate-in fade-in duration-300">
      {/* Title */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-100 dark:bg-zinc-800 text-emerald-700 dark:text-emerald-300 text-xs font-semibold mb-2">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Stage 2 of 2</span>
        </div>
        <h2 className="text-xl font-bold text-primary-color">
          Analyzing Your Plant
        </h2>
        <p className="text-xs text-secondary-color mt-1">
          Please wait while our AI examines your leaf specimen.
        </p>
      </div>

      {/* Center Image + Pulse Ring */}
      <div className="relative w-36 h-36 mx-auto flex items-center justify-center">
        <div className="absolute inset-0 rounded-full border-4 border-emerald-500/20 animate-ping" />
        <div className="absolute inset-0 rounded-full border-2 border-emerald-500/40 animate-pulse" />
        
        <div className="w-28 h-28 rounded-full overflow-hidden border-2 border-emerald-500 shadow-xl bg-surface-subtle relative z-10">
          <img
            src={imagePreviewUrl}
            alt="Analyzing leaf"
            className="w-full h-full object-cover animate-botanical-pulse"
          />
        </div>
      </div>

      {/* Stage Progression Checklist */}
      <div className="space-y-3 max-w-xs mx-auto text-left py-2">
        {stages.map((st, idx) => {
          const isDone = stage > idx;
          const isCurrent = stage === idx;

          return (
            <div
              key={idx}
              className={`flex items-center gap-3 transition-all duration-300 ${
                isDone
                  ? 'text-emerald-700 dark:text-emerald-400 font-medium'
                  : isCurrent
                  ? 'text-primary-color font-bold scale-[1.02]'
                  : 'text-muted-color opacity-50'
              }`}
            >
              {isDone ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
              ) : isCurrent ? (
                <Loader2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 animate-spin shrink-0" />
              ) : (
                <div className="w-4 h-4 rounded-full border border-subtle shrink-0" />
              )}
              <div className="truncate">
                <span className="text-xs">{st.label}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom Tip */}
      <div className="pt-4 border-t border-subtle text-xs text-muted-color">
        <span className="font-semibold text-emerald-600 dark:text-emerald-400">Model:</span>{' '}
        {modelName} • Computer Vision Transfer Learning
      </div>
    </div>
  );
};
