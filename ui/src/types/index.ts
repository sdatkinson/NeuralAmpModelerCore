export interface Model {
  name: string;
  url: string;
  default?: boolean;
}

export interface IR {
  name: string;
  url: string;
  mix?: number;
  gain?: number;
  default?: boolean;
}

export interface Input {
  name: string;
  url: string;
  default?: boolean;
}

export enum PREVIEW_MODE {
  MODEL = 'model',
  IR = 'ir',
}

// Utility type to ensure non-empty arrays
type NonEmptyArray<T> = [T, ...T[]];

export interface T3kPlayerProps {
  models?: NonEmptyArray<Model>;
  irs?: NonEmptyArray<IR>;
  inputs?: NonEmptyArray<Input>;
  isLoading?: boolean;
  previewMode?: PREVIEW_MODE;
  onPlay?: ({
    model,
    ir,
    input,
  }: {
    model: Model;
    ir: IR;
    input: Input;
  }) => void;
  onModelChange?: (model: Model) => void;
  onInputChange?: (input: Input) => void;
  onIrChange?: (ir: IR) => void;
  id?: string;
}

export interface T3kSlimPlayerProps extends T3kPlayerProps {
  getData: () => Promise<{
    model: Model;
    ir: IR;
    input: Input;
  }>;
  size?: number;
}
