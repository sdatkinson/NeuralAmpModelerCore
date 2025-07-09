import React, {
  ReactNode,
  useEffect,
  useRef,
  useState,
  useCallback,
} from 'react';
import { ChevronDown } from 'lucide-react';

interface Option {
  value: string | number;
  label: string;
}

interface SelectProps {
  options: Option[];
  defaultOption?: string | number;
  onChange?: (selected: string | number) => void | Promise<void>;
  label?: string;
  disabled?: boolean;
  backgroundColor?: string;
  infoModal?: ReactNode;
  value?: string | number;
}

export const Select = ({
  options,
  label,
  defaultOption,
  onChange,
  disabled = false,
  backgroundColor = 'bg-zinc-900',
  infoModal,
  value,
}: SelectProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedOption, setSelectedOption] = useState(
    defaultOption
      ? options.find(opt => opt.value === defaultOption) || options[0]
      : options[0]
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const [activeIndex, setActiveIndex] = useState<number>(-1);

  // Create a safe ID from the label
  const selectId = label ? label.toLowerCase().replace(/\s+/g, '-') : 'select';

  useEffect(() => {
    const handleOutsideClick = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
    }

    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, [isOpen]);

  const handleSelect = useCallback(
    (option: Option) => {
      setSelectedOption(option);
      setIsOpen(false);
      buttonRef.current?.focus();
      if (onChange) {
        onChange(option.value);
      }
    },
    [onChange]
  );

  useEffect(() => {
    let button = buttonRef.current;
    const listener = (e: KeyboardEvent) => {
      if (!isOpen) {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
          e.preventDefault();
          setIsOpen(true);
          setActiveIndex(
            options.findIndex(opt => opt.value === selectedOption.value)
          );
        }
        return;
      }

      switch (e.key) {
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (activeIndex >= 0 && activeIndex < options.length) {
            handleSelect(options[activeIndex]);
          }
          break;
        case 'ArrowDown':
          e.preventDefault();
          if (activeIndex < options.length - 1) {
            setActiveIndex(prev => prev + 1);
          }
          break;
        case 'ArrowUp':
          e.preventDefault();
          if (activeIndex > 0) {
            setActiveIndex(prev => prev - 1);
          }
          break;
        case 'Escape':
          setIsOpen(false);
          buttonRef.current?.focus();
          break;
        case 'Tab':
          setIsOpen(false);
          break;
      }
    };

    if (button) {
      button.addEventListener('keydown', listener);
    }

    return () => {
      if (button) {
        button.removeEventListener('keydown', listener);
      }
    };
  }, [isOpen, options, activeIndex, selectedOption?.value, handleSelect]);

  // Add effect for scrolling active item into view
  useEffect(() => {
    if (!isOpen || activeIndex < 0 || !listRef.current) return;

    const activeElement = document.getElementById(
      `${selectId}-option-${activeIndex}`
    );
    if (!activeElement) return;

    const listElement = listRef.current;
    const activeRect = activeElement.getBoundingClientRect();
    const listRect = listElement.getBoundingClientRect();

    if (activeRect.bottom > listRect.bottom) {
      listElement.scrollTop += activeRect.bottom - listRect.bottom;
    } else if (activeRect.top < listRect.top) {
      listElement.scrollTop -= listRect.top - activeRect.top;
    }
  }, [activeIndex, isOpen, selectId]);

  useEffect(() => {
    if (!value) return;
    if (value !== selectedOption?.value) {
      setSelectedOption(options.find(opt => opt.value === value) || options[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <div
      ref={containerRef}
      className={'inline-flex flex-1 flex-col gap-1 w-full'}
    >
      <div className={'flex justify-between items-end'}>
        {label && (
          <span className='text-sm text-zinc-400' id={`${selectId}-label`}>
            {label}
          </span>
        )}
        {!!infoModal && infoModal}
      </div>
      <div className='relative flex-1'>
        <button
          ref={buttonRef}
          className={`flex items-center justify-between w-full overflow-hidden px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent focus:outline-none ${disabled ? 'touch-none cursor-not-allowed' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
          disabled={disabled}
          aria-haspopup='listbox'
          aria-expanded={isOpen}
          aria-labelledby={`${selectId}-label`}
          aria-controls={`${selectId}-options`}
          id={`${selectId}-button`}
        >
          <span className={'text-ellipsis text-nowrap overflow-hidden min-w-0'}>
            {selectedOption?.label}
          </span>
          <ChevronDown className='flex-shrink-0' size={24} />
        </button>
        {isOpen && (
          <div
            className={`absolute z-10 w-full mt-1 ${backgroundColor} rounded-md shadow-lg`}
          >
            <ul
              ref={listRef}
              id={`${selectId}-options`}
              role='listbox'
              className='py-1 overflow-auto text-base rounded-md max-h-48 focus:outline-none border border-zinc-700'
              aria-label={label || 'Options'}
              aria-labelledby={`${selectId}-label`}
            >
              {options.map((option, index) => (
                <li
                  key={option.value + `${index}`}
                  id={`${selectId}-option-${index}`}
                  role='option'
                  aria-selected={selectedOption.value === option.value}
                  className={`text-ellipsis text-nowrap overflow-hidden cursor-default select-none relative py-2 pl-3 pr-9 ${
                    activeIndex === index ? 'bg-zinc-800' : 'hover:bg-zinc-800'
                  }`}
                  onClick={() => handleSelect(option)}
                  onMouseEnter={() => setActiveIndex(index)}
                >
                  {option.label}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
