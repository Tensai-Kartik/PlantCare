import React from 'react';
import { Sparkles, ArrowRight, CheckCircle2 } from 'lucide-react';
import { ExampleLeaf } from '../../types';

interface ExampleCardsProps {
  examples: ExampleLeaf[];
  onSelectExample: (exampleId: string) => void;
  isLoading?: boolean;
}

export const ExampleCards: React.FC<ExampleCardsProps> = ({
  examples,
  onSelectExample,
  isLoading = false
}) => {
  return (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-muted-color">
            Or try a verified sample leaf
          </h4>
        </div>
        <span className="text-[11px] text-muted-color">1-Click Live AI Test</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {examples.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelectExample(item.id)}
            disabled={isLoading}
            className="group relative flex flex-col text-left p-2.5 rounded-2xl bg-surface hover:bg-surface-elevated border border-subtle hover:border-emerald-500/50 custom-shadow transition-all duration-200 hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50"
          >
            {/* Thumbnail */}
            <div className="w-full h-24 rounded-xl overflow-hidden mb-2 bg-surface-subtle relative">
              <img
                src={item.image_url}
                alt={item.title}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                onError={(e) => {
                  // Fallback to stylized SVG placeholder if image path fails
                  const target = e.target as HTMLImageElement;
                  target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 24 24' fill='none' stroke='%2316a34a' stroke-width='2'%3E%3Cpath d='M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z'/%3E%3C/svg%3E";
                }}
              />
              {item.is_healthy ? (
                <span className="absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded-md bg-emerald-600/90 text-white text-[9px] font-bold flex items-center gap-0.5 shadow-xs">
                  <CheckCircle2 className="w-2.5 h-2.5" />
                  Healthy
                </span>
              ) : (
                <span className="absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded-md bg-amber-600/90 text-white text-[9px] font-bold shadow-xs">
                  Disease
                </span>
              )}
            </div>

            {/* Labels */}
            <div className="flex-1 flex flex-col justify-between">
              <div>
                <p className="text-[11px] font-bold text-primary-color leading-tight group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors line-clamp-1">
                  {item.title}
                </p>
                <p className="text-[10px] text-muted-color italic truncate mt-0.5">
                  {item.condition}
                </p>
              </div>

              <div className="mt-2 flex items-center justify-between text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                <span>Analyze</span>
                <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
