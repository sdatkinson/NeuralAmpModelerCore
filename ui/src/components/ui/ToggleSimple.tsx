import React from 'react';

interface ToggleSimpleProps {
  isChecked: boolean;
  onChange: (isChecked: boolean) => void;
  disabled?: boolean;
  label: string;
  ariaLabel?: string;
}

export const ToggleSimple = ({
  isChecked,
  onChange,
  disabled = false,
  label,
  ariaLabel,
}: ToggleSimpleProps) => {
  const handleToggle = () => {
    const newValue = !isChecked;
    onChange(newValue);
  };

  return (
    <label
      className={`flex flex-col gap-1 ${
        disabled ? 'touch-none cursor-not-allowed' : 'cursor-pointer'
      }`}
    >
      {label && <span className={`text-sm text-zinc-400`}>{label}</span>}
      <div className='relative'>
        <input
          type='checkbox'
          className='sr-only'
          checked={isChecked}
          onChange={handleToggle}
          disabled={disabled}
          aria-label={ariaLabel}
        />
        <div
          style={isChecked ? { backgroundColor: '#00D13B' } : {}}
          className={`w-10 h-6 bg-zinc-500 rounded-full shadow-inner transition-colors duration-300 ease-in-out`}
        ></div>
        <div
          className={`absolute w-4 h-4 bg-white rounded-full shadow top-1 left-1 transition-transform duration-300 ease-in-out ${
            isChecked ? 'transform translate-x-full' : ''
          }`}
        ></div>
      </div>
    </label>
  );
};
