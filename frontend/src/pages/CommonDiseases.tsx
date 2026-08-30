import React, { useState, useEffect } from 'react';
import { ShieldAlert, ChevronRight, CheckCircle2, BookOpen, Sparkles, Sprout } from 'lucide-react';
import { DiseaseInfo } from '../types';
import { fetchDiseases } from '../services/api';

interface CommonDiseasesProps {
  onOpenDiseaseInKB?: (diseaseId: string) => void;
}

export const CommonDiseases: React.FC<CommonDiseasesProps> = ({ onOpenDiseaseInKB }) => {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [selectedCrop, setSelectedCrop] = useState<string>('all');
  const [crops, setCrops] = useState<string[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    fetchDiseases().then((res) => {
      setDiseases(res.diseases);
      setCrops(res.plants);
    });
  }, []);

  const filtered = selectedCrop === 'all' 
    ? diseases.filter(d => !d.is_healthy)
    : diseases.filter(d => d.plant.toLowerCase() === selectedCrop.toLowerCase() && !d.is_healthy);

  const getSeverityBadge = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300 border-red-200 dark:border-red-900';
      case 'moderate':
        return 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border-amber-200 dark:border-amber-900';
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 border-blue-200 dark:border-blue-900';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-zinc-800 dark:text-gray-300 border-gray-200 dark:border-zinc-700';
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6 animate-in fade-in duration-300">
      <div>
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400 mb-1">
          <ShieldAlert className="w-4 h-4" />
          <span>Agricultural Threat Directory</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-black text-primary-color tracking-tight">
          Common Plant Diseases
        </h2>
        <p className="text-xs sm:text-sm text-secondary-color mt-1">
          Browse prevalent fungal, bacterial, and viral pathogens affecting major agricultural crops with symptoms and immediate action steps.
        </p>
      </div>

      {/* Crop Filter Pills */}
      <div className="bg-surface border border-subtle rounded-3xl p-4 sm:p-5 custom-shadow space-y-3">
        <div className="flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar text-xs">
          <span className="text-[11px] font-bold text-muted-color uppercase tracking-wider mr-1 shrink-0">
            Host Crop:
          </span>
          <button
            onClick={() => setSelectedCrop('all')}
            className={`px-3.5 py-1.5 rounded-xl font-medium whitespace-nowrap transition-all cursor-pointer ${
              selectedCrop === 'all'
                ? 'bg-emerald-600 text-white font-bold shadow-xs'
                : 'bg-surface-elevated border border-subtle text-secondary-color hover:text-primary-color'
            }`}
          >
            All Crops ({diseases.filter(d => !d.is_healthy).length})
          </button>
          {crops.map((crop) => {
            const count = diseases.filter(d => d.plant.toLowerCase() === crop.toLowerCase() && !d.is_healthy).length;
            return (
              <button
                key={crop}
                onClick={() => setSelectedCrop(crop)}
                className={`px-3.5 py-1.5 rounded-xl font-medium whitespace-nowrap transition-all cursor-pointer ${
                  selectedCrop === crop
                    ? 'bg-emerald-600 text-white font-bold shadow-xs'
                    : 'bg-surface-elevated border border-subtle text-secondary-color hover:text-primary-color'
                }`}
              >
                {crop} ({count})
              </button>
            );
          })}
        </div>
      </div>

      {/* Diseases List */}
      <div className="space-y-4">
        {filtered.map((d) => {
          const isExpanded = expandedId === d.id;
          const thumbUrl = d.image_url || `/examples/${d.id}.jpg`;
          return (
            <div
              key={d.id}
              className="bg-surface border border-subtle rounded-3xl p-5 sm:p-6 custom-shadow space-y-4 transition-all hover:border-emerald-500/40"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-start gap-4">
                  <div className="w-16 h-16 rounded-2xl overflow-hidden bg-surface-subtle border border-subtle shrink-0">
                    <img
                      src={thumbUrl}
                      alt={d.name}
                      className="w-full h-full object-cover"
                    />
                  </div>

                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300">
                        {d.plant}
                      </span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getSeverityBadge(d.severity)}`}>
                        {d.severity}
                      </span>
                    </div>
                    <h3 className="text-base sm:text-lg font-bold text-primary-color">{d.name}</h3>
                    {d.scientific_name && (
                      <p className="text-xs text-muted-color italic">{d.scientific_name}</p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 self-end sm:self-center">
                  {onOpenDiseaseInKB && (
                    <button
                      onClick={() => onOpenDiseaseInKB(d.id)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-50 dark:bg-zinc-800 hover:bg-emerald-100 text-emerald-700 dark:text-emerald-300 text-xs font-semibold border border-emerald-200 dark:border-zinc-700 transition-all cursor-pointer"
                    >
                      <BookOpen className="w-3.5 h-3.5" />
                      <span>Full Guide</span>
                    </button>
                  )}
                  <button
                    onClick={() => setExpandedId(isExpanded ? null : d.id)}
                    className="px-3.5 py-1.5 rounded-xl border border-subtle hover:bg-surface-elevated text-xs font-semibold text-primary-color transition-all cursor-pointer"
                  >
                    {isExpanded ? 'Hide Details' : 'View Symptoms & Care'}
                  </button>
                </div>
              </div>

              <p className="text-xs sm:text-sm text-secondary-color leading-relaxed">
                {d.description}
              </p>

              {isExpanded && (
                <div className="pt-4 border-t border-subtle grid grid-cols-1 md:grid-cols-2 gap-4 text-xs animate-in fade-in duration-200">
                  <div className="p-4 rounded-2xl bg-surface-elevated border border-subtle space-y-2">
                    <span className="font-bold text-primary-color flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                      Primary Diagnostic Symptoms:
                    </span>
                    <ul className="space-y-1.5 text-secondary-color">
                      {d.symptoms.map((s, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-emerald-600 font-bold">•</span>
                          <span>{s}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="p-4 rounded-2xl bg-surface-elevated border border-subtle space-y-2">
                    <span className="font-bold text-primary-color flex items-center gap-1.5">
                      <Sparkles className="w-4 h-4 text-amber-500" />
                      Immediate Action & Emergency Quarantine:
                    </span>
                    <ul className="space-y-1.5 text-secondary-color">
                      {d.treatment.immediate_steps.map((st, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-amber-600 font-bold">✓</span>
                          <span>{st}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
