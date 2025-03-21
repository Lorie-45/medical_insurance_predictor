import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

const ResultDisplay = ({ amount, loading }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Show animation when amount changes
    if (amount !== null || loading) {
      setVisible(true);
    }
  }, [amount, loading]);

  // Format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  if (!visible) return null;

  return (
    <div
      className={cn(
        'glass-card p-6 mt-6 overflow-hidden animate-slide-up',
        loading ? 'animate-pulse' : ''
      )}
    >
      <div className="w-full flex flex-col items-center justify-center">
        <div className="text-xs font-medium text-insurance-purple uppercase tracking-wider mb-1">
          Estimated Premium
        </div>
        {loading ? (
          <div className="h-16 flex items-center justify-center">
            <div className="w-8 h-8 border-4 border-insurance-purple/30 border-t-insurance-purple rounded-full animate-spin"></div>
          </div>
        ) : (
          <div className="flex flex-col items-center animate-fade-in">
            <div className="text-4xl md:text-5xl font-bold text-insurance-purple">
              {formatCurrency(amount || 0)}
            </div>
            <div className="text-sm text-muted-foreground mt-2">
              Annual Insurance Premium
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultDisplay;