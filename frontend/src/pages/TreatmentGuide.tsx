import React, { useState, useEffect } from 'react';
import { 
  Stethoscope, 
  ShieldAlert, 
  Sparkles, 
  Beaker, 
  CheckCircle2, 
  AlertTriangle, 
  BookOpen, 
  ChevronRight,
  Filter,
  Check
} from 'lucide-react';
import { DiseaseInfo } from '../types';
import { fetchDiseases } from '../services/api';

interface TreatmentGuideProps {
  onOpenDiseaseInKB?: (diseaseId: string) => void;
}

export const TreatmentGuide: React.FC<TreatmentGuideProps> = ({ onOpenDiseaseInKB }) => {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>('tomato_early_blight');

  useEffect(() => {
    fetchDiseases().then((res) => {
      setDiseases(res.diseases);
    });
  }, []);

  const currentDisease = diseases.find(d => d.id === selectedDiseaseId) || diseases[0];

  return (
    <div className="w-full max-w-5xl mx-auto space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400 mb-1">
          <Stethoscope className="w-4 h-4" />
          <span>Integrated Pest & Disease Management (IPM)</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-black text-primary-color tracking-tight">
          Plant Disease Treatment Guide
        </h2>
        <p className="text-xs sm:text-sm text-secondary-color mt-1">
          A structured agricultural framework for mitigating, managing, and treating foliar crop diseases responsibly with organic and conventional protocols.
        </p>
      </div>

      {/* Disease-Specific Protocol Interactive Selector */}
      <div className="bg-gradient-to-r from-emerald-500/10 via-teal-500/5 to-surface border border-emerald-500/30 rounded-3xl p-6 sm:p-7 custom-shadow space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-2xl bg-emerald-600 text-white shadow-md shadow-emerald-600/20">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400 block">
                Tailored Prescription Protocol
              </span>
              <h3 className="text-lg sm:text-xl font-bold text-primary-color">
                Specific Treatment Lookup by Condition
              </h3>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="disease-select" className="text-xs font-bold text-muted-color">Select Disease:</label>
            <select
              id="disease-select"
              value={selectedDiseaseId}
              onChange={(e) => setSelectedDiseaseId(e.target.value)}
              className="px-3.5 py-2 rounded-xl bg-surface border border-subtle text-xs font-semibold text-primary-color focus:outline-hidden focus:border-emerald-500 transition-all cursor-pointer"
            >
              {diseases.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.plant}: {d.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {currentDisease && (
          <div className="p-5 sm:p-6 rounded-2xl bg-surface border border-subtle space-y-5 animate-in fade-in duration-200">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-subtle">
              <div className="flex items-center gap-3.5">
                <div className="w-14 h-14 rounded-2xl overflow-hidden bg-surface-subtle border border-subtle shrink-0 shadow-xs">
                  <img
                    src={currentDisease.image_url || `/examples/${currentDisease.id}.jpg`}
                    alt={currentDisease.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300">
                      {currentDisease.plant}
                    </span>
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-surface-elevated border border-subtle text-secondary-color">
                      Severity: {currentDisease.severity}
                    </span>
                  </div>
                  <h4 className="text-base sm:text-lg font-black text-primary-color">{currentDisease.name}</h4>
                  {currentDisease.scientific_name && (
                    <span className="text-xs text-muted-color italic font-mono block">
                      Pathogen: {currentDisease.scientific_name}
                    </span>
                  )}
                </div>
              </div>
              {onOpenDiseaseInKB && (
                <button
                  onClick={() => onOpenDiseaseInKB(currentDisease.id)}
                  className="flex items-center gap-1.5 self-start sm:self-center px-3.5 py-1.5 rounded-xl bg-emerald-50 dark:bg-zinc-800 hover:bg-emerald-100 text-emerald-700 dark:text-emerald-300 text-xs font-bold border border-emerald-200 dark:border-zinc-700 transition-all cursor-pointer shadow-xs"
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  <span>Full Pathology Guide</span>
                </button>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              {/* Immediate Steps */}
              <div className="p-4 rounded-2xl bg-red-50/60 dark:bg-red-950/20 border border-red-200 dark:border-red-900/40 space-y-2">
                <span className="font-bold text-red-800 dark:text-red-300 flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4" />
                  Immediate Action:
                </span>
                <ul className="space-y-1 text-red-950 dark:text-red-200">
                  {currentDisease.treatment.immediate_steps.map((st, i) => (
                    <li key={i}>• {st}</li>
                  ))}
                </ul>
              </div>

              {/* Organic Options */}
              <div className="p-4 rounded-2xl bg-emerald-50/60 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/40 space-y-2">
                <span className="font-bold text-emerald-800 dark:text-emerald-300 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4" />
                  Organic Controls:
                </span>
                <ul className="space-y-1 text-emerald-950 dark:text-emerald-200">
                  {currentDisease.treatment.organic_options.map((st, i) => (
                    <li key={i}>✓ {st}</li>
                  ))}
                </ul>
              </div>

              {/* Conventional Options */}
              <div className="p-4 rounded-2xl bg-blue-50/60 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/40 space-y-2">
                <span className="font-bold text-blue-800 dark:text-blue-300 flex items-center gap-1.5">
                  <Beaker className="w-4 h-4" />
                  Conventional Treatments:
                </span>
                <ul className="space-y-1 text-blue-950 dark:text-blue-200">
                  {currentDisease.treatment.conventional_options.map((st, i) => (
                    <li key={i}>• {st}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Structured IPM Framework */}
      <div className="space-y-6">
        <h3 className="text-xl font-bold text-primary-color tracking-tight">
          Standard Integrated Pest Management (IPM) Pillars
        </h3>

        {/* Step 1: Immediate Quarantine */}
        <div className="bg-surface border border-subtle rounded-3xl p-6 custom-shadow space-y-3">
          <div className="flex items-center gap-2.5 text-red-600 dark:text-red-400 font-bold text-base">
            <ShieldAlert className="w-5 h-5" />
            <span>1. Immediate Emergency Quarantine & Sanitization Steps</span>
          </div>
          <p className="text-xs sm:text-sm text-secondary-color leading-relaxed">
            When fungal spots, bacterial lesions, or viral mottling first appear on foliage, act immediately to prevent mechanical and spore transmission:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-xs">
            <div className="p-4 rounded-2xl bg-red-50/50 dark:bg-zinc-900 border border-red-200 dark:border-zinc-800 space-y-1">
              <span className="font-bold text-red-900 dark:text-red-300 block">Sanitized Pruning</span>
              <p className="text-red-950 dark:text-zinc-200 leading-relaxed">
                Prune off infected lower leaves immediately using sharp shears. Dip cutting blades in 70% isopropyl alcohol or 10% bleach between every cut to prevent cross-inoculation.
              </p>
            </div>
            <div className="p-4 rounded-2xl bg-red-50/50 dark:bg-zinc-900 border border-red-200 dark:border-zinc-800 space-y-1">
              <span className="font-bold text-red-900 dark:text-red-300 block">Secure Disposal</span>
              <p className="text-red-950 dark:text-zinc-200 leading-relaxed">
                Bag diseased plant tissue immediately and discard in municipal trash. Never add infected plant parts to domestic compost piles as fungal resting spores survive low-temperature composting.
              </p>
            </div>
          </div>
        </div>

        {/* Step 2: Organic Remedies */}
        <div className="bg-surface border border-subtle rounded-3xl p-6 custom-shadow space-y-3">
          <div className="flex items-center gap-2.5 text-emerald-600 dark:text-emerald-400 font-bold text-base">
            <Sparkles className="w-5 h-5" />
            <span>2. Organic & Biological Controls</span>
          </div>
          <p className="text-xs sm:text-sm text-secondary-color leading-relaxed">
            Low-impact biological solutions suppress pathogens without harming pollinating insects or creating chemical residue:
          </p>
          <div className="space-y-2.5 pt-2 text-xs">
            <div className="p-3.5 rounded-2xl bg-surface-elevated flex items-start gap-3 border border-subtle">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-primary-color">Bacillus subtilis / Biofungicides:</strong>
                <p className="text-secondary-color mt-0.5">
                  Beneficial bacterial strains that competitively colonize plant surfaces and synthesize antimicrobial peptides inhibiting spore germination (Early Blight, Powdery Mildew, Botrytis).
                </p>
              </div>
            </div>
            <div className="p-3.5 rounded-2xl bg-surface-elevated flex items-start gap-3 border border-subtle">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-primary-color">Copper Octanoate (Fixed Copper Soap) & Bordeaux Mixture:</strong>
                <p className="text-secondary-color mt-0.5">
                  Fixed copper forms a protective barrier that denatures fungal and bacterial proteins upon contact.
                </p>
              </div>
            </div>
            <div className="p-3.5 rounded-2xl bg-surface-elevated flex items-start gap-3 border border-subtle">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-primary-color">Potassium Bicarbonate:</strong>
                <p className="text-secondary-color mt-0.5">
                  Alters leaf surface pH to disrupt active fungal mycelial growth without burning delicate foliage.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Step 3: Conventional Fungicide Principles */}
        <div className="bg-surface border border-subtle rounded-3xl p-6 custom-shadow space-y-3">
          <div className="flex items-center gap-2.5 text-blue-600 dark:text-blue-400 font-bold text-base">
            <Beaker className="w-5 h-5" />
            <span>3. Conventional Chemical Treatments & FRAC Rotation</span>
          </div>
          <p className="text-xs sm:text-sm text-secondary-color leading-relaxed">
            For severe agricultural outbreaks, synthetic systemic and protectant fungicides provide curative control:
          </p>
          <div className="p-4 rounded-2xl bg-blue-50/50 dark:bg-zinc-900 border border-blue-200 dark:border-zinc-800 text-xs space-y-2">
            <span className="font-bold text-blue-950 dark:text-blue-200 block">Critical Resistance Management Rules:</span>
            <ul className="space-y-1.5 text-blue-900 dark:text-zinc-200">
              <li>• <strong>Rotate Modes of Action:</strong> Alternate FRAC chemical groups (e.g. Group 11 Strobilurins vs Group 3 Triazoles vs Group M multi-site protectants like Mancozeb or Chlorothalonil) to prevent pathogen mutations.</li>
              <li>• <strong>Observe Pre-Harvest Intervals (PHI):</strong> Strictly follow the mandatory days between pesticide application and harvesting edible produce.</li>
              <li>• <strong>Canopy Penetration:</strong> Ensure spray droplets reach lower leaf undersides where high humidity promotes fungal sporulation.</li>
            </ul>
          </div>
        </div>

        {/* Regulatory Disclaimer */}
        <div className="p-4 rounded-2xl bg-amber-50 dark:bg-zinc-900 border border-amber-200 dark:border-zinc-800 flex items-start gap-3 text-xs text-amber-950 dark:text-zinc-200">
          <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <p>
            Always adhere to local agricultural regulations and read all chemical pesticide labels completely prior to mixing and application. Consult your regional university agricultural extension service for registered fungicides and target treatment dates.
          </p>
        </div>
      </div>
    </div>
  );
};
