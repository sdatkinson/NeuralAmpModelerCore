import React from 'react';

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string> {
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  size?: 'sm' | 'md';
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  disabled = false,
  size = 'md',
}: SegmentedControlProps<T>) {
  const sizeClasses = size === 'sm'
    ? 'px-2.5 py-1 text-xs'
    : 'px-4 py-1.5 text-sm';

  return (
    <div
      className={`inline-flex rounded-lg bg-zinc-800 p-1 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      {options.map(option => {
        const isSelected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            disabled={disabled}
            onClick={() => !disabled && onChange(option.value)}
            className={`
              ${sizeClasses} font-medium rounded-md transition-all duration-150
              ${isSelected
                ? 'bg-zinc-600 text-white shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
              }
              ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export default SegmentedControl;
