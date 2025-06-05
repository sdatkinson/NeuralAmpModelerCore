export interface Model {
  name: string;
  model_url: string;
}

export interface IR {
  name: string;
  ir_url: string;
  mix?: number;
  gain?: number;
}

export interface Input {
  name: string;
  input_url: string;
}

export interface T3kPlayerProps {
  models: Model[];
  irs: IR[];
  inputs: Input[];
  isLoading?: boolean;
}