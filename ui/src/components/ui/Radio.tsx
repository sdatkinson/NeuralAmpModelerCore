import React from 'react';

export interface RadioProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type' | 'size'> {
  /** Whether the radio is selected */
  checked: boolean;
  /** Change handler */
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  /** Value for the radio input */
  value: string;
  /** Name for the radio group */
  name: string;
  /** Optional label - when provided, clicking the label toggles the radio */
  label?: React.ReactNode;
  /** Additional class for the wrapper */
  className?: string;
}

/**
 * A radio button matching the design: selected = white inner dot with dark grey
 * ring; unselected = thin light grey outline, transparent interior.
 */
export const Radio: React.FC<RadioProps> = ({
  checked,
  onChange,
  value,
  name,
  label,
  className = '',
  disabled,
  ...props
}) => {
  const content = (
    <>
      <span
        className={`
          flex items-center justify-center w-5 h-5 rounded-full flex-shrink-0
          transition-colors duration-150
          ${
            checked
              ? 'border border-white bg-transparent'
              : 'border border-zinc-500 bg-transparent'
          }
          ${disabled ? 'opacity-50' : ''}
        `}
        aria-hidden
      >
        {checked && <span className='w-3 h-3 rounded-full bg-white' />}
      </span>
      <input
        type='radio'
        name={name}
        value={value}
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        className='sr-only'
        {...props}
      />
      {label != null && <span className='text-sm text-zinc-300'>{label}</span>}
    </>
  );

  if (label != null) {
    return (
      <label
        className={`flex items-center gap-3 cursor-pointer ${disabled ? 'cursor-not-allowed' : ''} ${className}`}
      >
        {content}
      </label>
    );
  }

  return <div className={`flex items-center ${className}`}>{content}</div>;
};

Radio.displayName = 'Radio';
