import React from 'react';
import { 
  Sparkles, 
  ShieldCheck, 
  Zap, 
  Eye, 
  Lock, 
  ArrowRight 
} from 'lucide-react';
import { LeafDropzone } from '../components/upload/LeafDropzone';
import { ExampleCards } from '../components/upload/ExampleCards';
import { ExampleLeaf } from '../types';

interface DashboardProps {
  onImageSelected: (file: File) => void;
  onSelectExample: (exampleId: string) => void;
  examples: ExampleLeaf[];
  isLoading?: boolean;
}

export const Dashboard: React.FC<DashboardProps> = ({
  onImageSelected,
  onSelectExample,
  examples,
  isLoading = false
}) => {
  // Determine greeting by time of day
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning!' : hour < 18 ? 'Good afternoon!' : 'Good evening!';

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 animate-in fade-in duration-300">
      {/* Hero Welcome Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400 mb-1">
          <span>{greeting} 🌱</span>
        </div>
        <h2 className="text-2xl sm:text-4xl font-black text-primary-color tracking-tight leading-tight">
          Let's keep your <span className="text-emerald-600 dark:text-emerald-400 underline decoration-emerald-500/30">plants healthy.</span>
        </h2>
        <p className="text-sm sm:text-base text-secondary-color mt-2 max-w-2xl leading-relaxed">
          Upload a leaf image and our AI will identify the condition, visualize model attention with Grad-CAM, and provide practical care solutions.
        </p>
      </div>

      {/* Main Upload Area */}
      <div className="bg-surface border border-subtle rounded-3xl p-6 sm:p-8 custom-shadow">
        <LeafDropzone onImageSelected={onImageSelected} isLoading={isLoading} />
      </div>

      {/* Verified Sample Examples Section */}
      <div className="pt-2">
        <ExampleCards
          examples={examples}
          onSelectExample={onSelectExample}
          isLoading={isLoading}
        />
      </div>

      {/* Feature Capability Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 pt-4 border-t border-subtle">
        <div className="p-4 rounded-2xl bg-surface border border-subtle custom-shadow flex items-start gap-3">
          <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-primary-color">High Accuracy</h4>
            <p className="text-[11px] text-muted-color mt-0.5">Transfer learning on 21 agricultural classes</p>
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-surface border border-subtle custom-shadow flex items-start gap-3">
          <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-primary-color">Fast & Reliable</h4>
            <p className="text-[11px] text-muted-color mt-0.5">Optimized lightweight CPU inference</p>
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-surface border border-subtle custom-shadow flex items-start gap-3">
          <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">
            <Eye className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-primary-color">Explainable AI</h4>
            <p className="text-[11px] text-muted-color mt-0.5">Grad-CAM heatmaps reveal model focus</p>
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-surface border border-subtle custom-shadow flex items-start gap-3">
          <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">
            <Lock className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-primary-color">Privacy-First</h4>
            <p className="text-[11px] text-muted-color mt-0.5">Images are processed in memory and not stored</p>
          </div>
        </div>
      </div>
    </div>
  );
};
