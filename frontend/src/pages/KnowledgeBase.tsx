import React, { useState, useEffect } from 'react';
import { 
  Search, 
  BookOpen, 
  X, 
  CheckCircle2, 
  ShieldAlert, 
  Sparkles, 
  ChevronRight, 
  Layers, 
  Activity, 
  Sprout, 
  AlertTriangle, 
  Beaker, 
  Check, 
  Info,
  ExternalLink,
  SlidersHorizontal
} from 'lucide-react';
import { DiseaseInfo } from '../types';
import { fetchDiseases } from '../services/api';

interface KnowledgeBaseProps {
  initialDiseaseId?: string | null;
  onClearInitialDisease?: () => void;
}

type ModalTabType = 'overview' | 'symptoms' | 'causes' | 'treatment' | 'prevention' | 'notes';

export const KnowledgeBase: React.FC<KnowledgeBaseProps> = ({
  initialDiseaseId,
  onClearInitialDisease
}) => {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [plants, setPlants] = useState<string[]>([]);
  const [selectedPlant, setSelectedPlant] = useState<string>('all');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedDiseaseModal, setSelectedDiseaseModal] = useState<DiseaseInfo | null>(null);
  const [activeModalTab, setActiveModalTab] = useState<ModalTabType>('overview');
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    loadDiseases();
  }, [selectedPlant, selectedSeverity, searchQuery]);

  // If initialDiseaseId is provided, open that modal automatically once loaded
  useEffect(() => {
    if (initialDiseaseId && diseases.length > 0) {
      const match = diseases.find(d => d.id === initialDiseaseId || d.id.toLowerCase() === initialDiseaseId.toLowerCase());
      if (match) {
        setSelectedDiseaseModal(match);
        setActiveModalTab('overview');
      }
    }
  }, [initialDiseaseId, diseases]);

  const loadDiseases = async () => {
    setIsLoading(true);
    try {
      const res = await fetchDiseases({
        plant: selectedPlant,
        severity: selectedSeverity,
        q: searchQuery
      });
      setDiseases(res.diseases);
      if (res.plants && res.plants.length > 0) {
        setPlants(res.plants);
      }
    } catch (err) {
      console.error('Failed to load diseases:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCloseModal = () => {
    setSelectedDiseaseModal(null);
    if (onClearInitialDisease) {
      onClearInitialDisease();
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'healthy':
        return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border-emerald-200 dark:border-emerald-900';
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 border-blue-200 dark:border-blue-900';
      case 'moderate':
        return 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border-amber-200 dark:border-amber-900';
      case 'high':
      case 'critical':
        return 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300 border-red-200 dark:border-red-900';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-zinc-800 dark:text-gray-300 border-gray-200 dark:border-zinc-700';
    }
  };

  const modalTabs: { id: ModalTabType; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <Info className="w-3.5 h-3.5" /> },
    { id: 'symptoms', label: 'Symptoms', icon: <CheckCircle2 className="w-3.5 h-3.5" /> },
    { id: 'causes', label: 'Causes', icon: <Activity className="w-3.5 h-3.5" /> },
    { id: 'treatment', label: 'Treatment Guide', icon: <Sparkles className="w-3.5 h-3.5" /> },
    { id: 'prevention', label: 'Prevention', icon: <Check className="w-3.5 h-3.5" /> },
    { id: 'notes', label: 'Advisory Notes', icon: <AlertTriangle className="w-3.5 h-3.5" /> },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400 mb-1">
            <BookOpen className="w-4 h-4" />
            <span>Botanical Pathology Library</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-black text-primary-color tracking-tight">
            Plant Disease Knowledge Base
          </h2>
          <p className="text-xs sm:text-sm text-secondary-color mt-1">
            Explore clinically verified pathology profiles, diagnostic symptoms, environmental causes, and integrated management guides for 21 agricultural classes.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto text-xs px-3.5 py-1.5 rounded-xl bg-surface border border-subtle text-secondary-color">
          <Sprout className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          <span className="font-semibold text-primary-color">{diseases.length}</span> Conditions Cataloged
        </div>
      </div>

      {/* Search & Filter Controls */}
      <div className="bg-surface border border-subtle rounded-3xl p-5 custom-shadow space-y-4">
        {/* Search Bar */}
        <div className="relative">
          <Search className="w-4 h-4 text-muted-color absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search across 21 diseases by name, pathogen, host crop, or symptom..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-10 py-2.5 rounded-xl bg-surface-elevated border border-subtle text-xs sm:text-sm text-primary-color placeholder:text-muted-color focus:outline-hidden focus:border-emerald-500 transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-color hover:text-primary-color p-1"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Plant Category Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar text-xs">
          <span className="text-[11px] font-bold text-muted-color uppercase tracking-wider mr-1 shrink-0">
            Host Crop:
          </span>
          <button
            onClick={() => setSelectedPlant('all')}
            className={`px-3 py-1.5 rounded-xl font-medium whitespace-nowrap transition-all cursor-pointer ${
              selectedPlant === 'all'
                ? 'bg-emerald-600 text-white font-bold shadow-xs'
                : 'bg-surface-elevated text-secondary-color hover:text-primary-color border border-subtle'
            }`}
          >
            All Crops ({diseases.length})
          </button>
          {plants.map((plant) => (
            <button
              key={plant}
              onClick={() => setSelectedPlant(plant)}
              className={`px-3 py-1.5 rounded-xl font-medium whitespace-nowrap transition-all cursor-pointer ${
                selectedPlant === plant
                  ? 'bg-emerald-600 text-white font-bold shadow-xs'
                  : 'bg-surface-elevated text-secondary-color hover:text-primary-color border border-subtle'
              }`}
            >
              {plant}
            </button>
          ))}
        </div>

        {/* Severity Filter Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar text-xs pt-1 border-t border-subtle">
          <span className="text-[11px] font-bold text-muted-color uppercase tracking-wider mr-1 shrink-0">
            Severity:
          </span>
          {['all', 'Healthy', 'Moderate', 'High', 'Critical'].map((sev) => (
            <button
              key={sev}
              onClick={() => setSelectedSeverity(sev)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                selectedSeverity === sev
                  ? 'bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-300 font-bold border border-emerald-300 dark:border-emerald-800'
                  : 'bg-surface-elevated text-muted-color hover:text-primary-color border border-transparent'
              }`}
            >
              {sev === 'all' ? 'All Severities' : sev}
            </button>
          ))}
        </div>
      </div>

      {/* Disease Cards Grid */}
      {isLoading ? (
        <div className="text-center py-16 text-sm text-muted-color">
          <div className="w-8 h-8 rounded-full border-2 border-emerald-500 border-t-transparent animate-spin mx-auto mb-3" />
          <span>Searching botanical pathology database...</span>
        </div>
      ) : diseases.length === 0 ? (
        <div className="text-center py-12 bg-surface rounded-3xl border border-subtle p-8 custom-shadow space-y-2">
          <p className="text-base font-bold text-primary-color">No matching plant diseases found</p>
          <p className="text-xs text-muted-color">Try clearing your filters or changing your search terms.</p>
          <button
            onClick={() => { setSelectedPlant('all'); setSelectedSeverity('all'); setSearchQuery(''); }}
            className="mt-3 px-4 py-2 rounded-xl bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 transition-colors cursor-pointer"
          >
            Reset Filters
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {diseases.map((d) => {
            const thumbUrl = d.image_url || `/examples/${d.id}.jpg`;
            return (
              <div
                key={d.id}
                onClick={() => { setSelectedDiseaseModal(d); setActiveModalTab('overview'); }}
                className="group flex flex-col justify-between rounded-3xl bg-surface hover:bg-surface-elevated border border-subtle hover:border-emerald-500/50 custom-shadow transition-all duration-200 cursor-pointer hover:-translate-y-1 overflow-hidden"
              >
                {/* Thumbnail Image Header */}
                <div className="relative h-44 w-full bg-surface-subtle overflow-hidden border-b border-subtle">
                  <img
                    src={thumbUrl}
                    alt={d.name}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    onError={(e) => {
                      // Fallback placeholder if image not loaded
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  <div className="absolute top-3 left-3 flex items-center gap-1.5">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-xs text-white">
                      {d.plant}
                    </span>
                  </div>
                  <div className="absolute top-3 right-3">
                    <span className={`text-[10px] font-bold px-2.5 py-1 rounded-lg border backdrop-blur-xs shadow-xs ${getSeverityBadge(d.severity)}`}>
                      {d.severity}
                    </span>
                  </div>
                </div>

                {/* Card Body */}
                <div className="p-5 flex-1 flex flex-col justify-between">
                  <div>
                    <h3 className="font-bold text-base text-primary-color group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors leading-snug">
                      {d.name}
                    </h3>
                    {d.scientific_name && (
                      <p className="text-xs text-muted-color italic mt-0.5 line-clamp-1">
                        {d.scientific_name}
                      </p>
                    )}

                    <p className="text-xs text-secondary-color mt-2.5 line-clamp-2 leading-relaxed">
                      {d.description}
                    </p>

                    {/* Top Symptom Teaser */}
                    {d.symptoms.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-subtle space-y-1">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-color">
                          Primary Symptom:
                        </span>
                        <p className="text-xs text-secondary-color truncate">
                          • {d.symptoms[0]}
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 pt-3 flex items-center justify-between text-xs font-semibold text-emerald-600 dark:text-emerald-400 border-t border-subtle">
                    <span>Explore Clinical Guide</span>
                    <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Disease Detail Modal */}
      {selectedDiseaseModal && (
        <div className="fixed inset-0 z-50 bg-black/65 backdrop-blur-xs flex items-center justify-center p-3 sm:p-5 animate-in fade-in duration-200">
          <div className="bg-surface border border-subtle rounded-3xl max-w-3xl w-full max-h-[90vh] overflow-y-auto custom-shadow flex flex-col relative animate-in zoom-in-95 duration-200">
            {/* Modal Header Bar with Close Button */}
            <div className="sticky top-0 z-20 bg-surface/95 backdrop-blur-md border-b border-subtle px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold px-2.5 py-1 rounded-lg bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                  {selectedDiseaseModal.plant}
                </span>
                <span className={`text-xs font-bold px-2.5 py-1 rounded-lg border ${getSeverityBadge(selectedDiseaseModal.severity)}`}>
                  {selectedDiseaseModal.severity}
                </span>
              </div>

              <button
                onClick={handleCloseModal}
                className="p-2 rounded-xl bg-surface-elevated hover:bg-surface text-secondary-color hover:text-primary-color border border-subtle transition-all cursor-pointer"
                title="Close Guide"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 sm:p-8 space-y-6">
              {/* Title & Specimen Header */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 items-start">
                <div className="sm:col-span-2 space-y-1">
                  <h2 className="text-xl sm:text-2xl font-black text-primary-color tracking-tight">
                    {selectedDiseaseModal.name}
                  </h2>
                  {selectedDiseaseModal.scientific_name && (
                    <p className="text-xs sm:text-sm text-emerald-600 dark:text-emerald-400 font-mono italic">
                      Pathogen: {selectedDiseaseModal.scientific_name}
                    </p>
                  )}
                  <p className="text-xs sm:text-sm text-secondary-color leading-relaxed pt-2">
                    {selectedDiseaseModal.description}
                  </p>
                </div>

                <div className="sm:col-span-1 rounded-2xl overflow-hidden bg-surface-subtle border border-subtle h-36 relative">
                  <img
                    src={selectedDiseaseModal.image_url || `/examples/${selectedDiseaseModal.id}.jpg`}
                    alt={selectedDiseaseModal.name}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute bottom-1.5 left-1.5 px-2 py-0.5 rounded-md bg-black/70 backdrop-blur-xs text-[10px] text-white">
                    Specimen
                  </div>
                </div>
              </div>

              {/* Modal Tabs Bar */}
              <div className="flex border-b border-subtle overflow-x-auto no-scrollbar gap-1 sm:gap-2 pt-2">
                {modalTabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveModalTab(tab.id)}
                    className={`flex items-center gap-1.5 pb-2.5 px-3 text-xs font-semibold whitespace-nowrap transition-all border-b-2 cursor-pointer ${
                      activeModalTab === tab.id
                        ? 'border-emerald-600 text-emerald-600 dark:border-emerald-400 dark:text-emerald-400 font-bold'
                        : 'border-transparent text-muted-color hover:text-primary-color'
                    }`}
                  >
                    {tab.icon}
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>

              {/* Tab 1: Overview */}
              {activeModalTab === 'overview' && (
                <div className="space-y-4 animate-in fade-in duration-150">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="p-3.5 rounded-2xl bg-surface-elevated border border-subtle">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-muted-color block mb-1">
                        Host Crop
                      </span>
                      <span className="text-sm font-bold text-primary-color">{selectedDiseaseModal.plant}</span>
                    </div>

                    <div className="p-3.5 rounded-2xl bg-surface-elevated border border-subtle">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-muted-color block mb-1">
                        Severity Classification
                      </span>
                      <span className={`inline-block px-2 py-0.5 rounded-md text-xs font-bold border ${getSeverityBadge(selectedDiseaseModal.severity)}`}>
                        {selectedDiseaseModal.severity}
                      </span>
                    </div>

                    <div className="p-3.5 rounded-2xl bg-surface-elevated border border-subtle">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-muted-color block mb-1">
                        Transmission Vector
                      </span>
                      <span className="text-xs text-secondary-color">
                        {selectedDiseaseModal.spread || 'Fungal spores / rain splash / wind'}
                      </span>
                    </div>
                  </div>

                  <div className="p-4 rounded-2xl bg-surface-elevated border border-subtle space-y-2">
                    <span className="text-xs font-bold text-primary-color block">
                      Agronomic Impact Summary:
                    </span>
                    <p className="text-xs text-secondary-color leading-relaxed">
                      {selectedDiseaseModal.is_healthy
                        ? 'This plant shows vigorous, healthy botanical tissue without visible foliar lesions, chlorosis, or necrotic decay.'
                        : `This condition causes foliar lesions and stress to ${selectedDiseaseModal.plant} crops, impacting photosynthetic capacity and harvest yield if left unchecked.`
                      }
                    </p>
                  </div>
                </div>
              )}

              {/* Tab 2: Symptoms */}
              {activeModalTab === 'symptoms' && (
                <div className="space-y-3 animate-in fade-in duration-150">
                  <h4 className="font-bold text-xs uppercase tracking-wider text-muted-color">
                    Diagnostic Visual Symptoms ({selectedDiseaseModal.symptoms.length}):
                  </h4>
                  <div className="space-y-2">
                    {selectedDiseaseModal.symptoms.map((s, i) => (
                      <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-surface-elevated border border-subtle text-xs text-secondary-color">
                        <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                        <span className="leading-relaxed">{s}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tab 3: Causes */}
              {activeModalTab === 'causes' && (
                <div className="space-y-3 animate-in fade-in duration-150">
                  <h4 className="font-bold text-xs uppercase tracking-wider text-muted-color">
                    Pathogenic & Environmental Drivers:
                  </h4>
                  <div className="space-y-2">
                    {selectedDiseaseModal.causes.map((c, i) => (
                      <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-surface-elevated border border-subtle text-xs text-secondary-color">
                        <div className="w-2 h-2 rounded-full bg-amber-500 shrink-0 mt-1.5" />
                        <span className="leading-relaxed">{c}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tab 4: Treatment */}
              {activeModalTab === 'treatment' && (
                <div className="space-y-4 animate-in fade-in duration-150">
                  {/* Immediate Emergency Steps */}
                  {selectedDiseaseModal.treatment.immediate_steps.length > 0 && (
                    <div className="p-4 rounded-2xl bg-red-50/70 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50 space-y-2 text-xs">
                      <span className="font-bold text-red-800 dark:text-red-300 flex items-center gap-1.5">
                        <ShieldAlert className="w-4 h-4" />
                        1. Immediate Emergency Quarantine & Sanitation:
                      </span>
                      <ul className="space-y-1 text-red-950 dark:text-red-200">
                        {selectedDiseaseModal.treatment.immediate_steps.map((st, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="font-bold">•</span>
                            <span>{st}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Organic Options */}
                  {selectedDiseaseModal.treatment.organic_options.length > 0 && (
                    <div className="p-4 rounded-2xl bg-emerald-50/70 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/50 space-y-2 text-xs">
                      <span className="font-bold text-emerald-800 dark:text-emerald-300 flex items-center gap-1.5">
                        <Sparkles className="w-4 h-4" />
                        2. Organic & Bio-Fungicide Solutions:
                      </span>
                      <ul className="space-y-1 text-emerald-950 dark:text-emerald-200">
                        {selectedDiseaseModal.treatment.organic_options.map((st, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="font-bold">✓</span>
                            <span>{st}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Conventional Options */}
                  {selectedDiseaseModal.treatment.conventional_options.length > 0 && (
                    <div className="p-4 rounded-2xl bg-blue-50/70 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/50 space-y-2 text-xs">
                      <span className="font-bold text-blue-800 dark:text-blue-300 flex items-center gap-1.5">
                        <Beaker className="w-4 h-4" />
                        3. Conventional Treatments & Fungicide Rotation:
                      </span>
                      <ul className="space-y-1 text-blue-950 dark:text-blue-200">
                        {selectedDiseaseModal.treatment.conventional_options.map((st, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="font-bold">•</span>
                            <span>{st}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 5: Prevention */}
              {activeModalTab === 'prevention' && (
                <div className="space-y-3 animate-in fade-in duration-150">
                  <h4 className="font-bold text-xs uppercase tracking-wider text-muted-color">
                    Cultural Prevention & Field Hygiene Practices:
                  </h4>
                  <div className="space-y-2">
                    {selectedDiseaseModal.prevention.map((p, i) => (
                      <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-surface-elevated border border-subtle text-xs text-secondary-color">
                        <Check className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                        <span className="leading-relaxed">{p}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tab 6: Notes */}
              {activeModalTab === 'notes' && (
                <div className="space-y-4 animate-in fade-in duration-150">
                  <div className="p-4 rounded-2xl bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/50 space-y-2 text-xs">
                    <span className="font-bold text-amber-800 dark:text-amber-300 flex items-center gap-1.5">
                      <AlertTriangle className="w-4 h-4" />
                      Important Agricultural & Regulatory Advisory:
                    </span>
                    <div className="space-y-1.5 text-amber-950 dark:text-amber-200">
                      {selectedDiseaseModal.important_notes.map((note, i) => (
                        <p key={i}>• {note}</p>
                      ))}
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-surface-elevated border border-subtle text-[11px] text-muted-color leading-relaxed">
                    <strong>Notice:</strong> Information provided is for educational decision support. Always consult registered chemical labels and local university extension services for commercial agriculture.
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
