import React from 'react';
import { Sprout, Droplets, Wind, RotateCw, Sparkles, CheckCircle2 } from 'lucide-react';

export const TipsPrevention: React.FC = () => {
  const tips = [
    {
      icon: <Droplets className="w-5 h-5 text-blue-500" />,
      title: 'Watering & Leaf Wetness Management',
      desc: 'Most foliar fungal spores require 6 to 12 hours of continuous leaf wetness to germinate. Always water at ground level using drip irrigation or soaker hoses in the early morning so any incidental splash dries quickly under daytime sunshine.'
    },
    {
      icon: <Wind className="w-5 h-5 text-teal-500" />,
      title: 'Airflow, Spacing & Trellising',
      desc: 'Dense plant canopies trap stagnant humid air. Maintain recommended crop spacing (at least 24-36 inches for tomatoes and peppers) and prune lower suckers to ensure light penetration and continuous breeze drying.'
    },
    {
      icon: <RotateCw className="w-5 h-5 text-amber-500" />,
      title: 'Strict 3- to 4-Year Crop Rotation',
      desc: 'Soil-borne pathogens (such as Alternaria and Xanthomonas) overwinter in soil residue. Never plant crops from the same botanical family (e.g., Solanaceae: Tomatoes, Potatoes, Peppers, Eggplants) in the exact same garden bed consecutive years.'
    },
    {
      icon: <Sprout className="w-5 h-5 text-emerald-500" />,
      title: 'Mulch Barrier Protection',
      desc: 'Apply a 2-3 inch layer of clean straw, wood chips, or plastic mulch around the base of vegetable plants. Mulch forms a physical shield that prevents rain and irrigation droplets from splashing fungal spores from the soil onto lower foliage.'
    },
    {
      icon: <Sparkles className="w-5 h-5 text-purple-500" />,
      title: 'Sanitation & Tool Sterilization',
      desc: 'Disinfect pruning shears between plants using 70% isopropyl alcohol. In autumn, clean and remove all spent plant debris from the garden floor to eliminate overwintering fungal spore reservoirs.'
    }
  ];

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-in fade-in duration-300">
      <div>
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400 mb-1">
          <Sprout className="w-4 h-4" />
          <span>Cultural Best Practices</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-black text-primary-color tracking-tight">
          Plant Health & Disease Prevention
        </h2>
        <p className="text-xs sm:text-sm text-secondary-color mt-1">
          Proactive cultural techniques to build plant immune resilience and prevent disease outbreaks before they start.
        </p>
      </div>

      {/* Prevention Cards Grid */}
      <div className="space-y-4">
        {tips.map((tip, idx) => (
          <div
            key={idx}
            className="p-5 sm:p-6 rounded-3xl bg-surface border border-subtle custom-shadow flex items-start gap-4"
          >
            <div className="p-3 rounded-2xl bg-surface-elevated shrink-0 shadow-inner">
              {tip.icon}
            </div>
            <div className="space-y-1">
              <h3 className="font-bold text-base text-primary-color">{tip.title}</h3>
              <p className="text-xs sm:text-sm text-secondary-color leading-relaxed">
                {tip.desc}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Summary Checklist */}
      <div className="p-6 rounded-3xl bg-emerald-50/60 dark:bg-zinc-900 border border-emerald-200/60 dark:border-zinc-800 space-y-3">
        <h3 className="font-bold text-sm text-emerald-950 dark:text-zinc-100">
          The Healthy Garden Checklist
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-emerald-900 dark:text-zinc-300">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
            <span>Water early morning at the roots</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
            <span>Apply 2-3" organic mulch barrier</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
            <span>Stake & trellis vertical vines</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
            <span>Disinfect tools between plants</span>
          </div>
        </div>
      </div>
    </div>
  );
};
