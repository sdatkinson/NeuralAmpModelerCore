const inputs = [
  {
    name: 'Brit - Guitar',
    url: '/inputs/Brit - Guitar.wav',
    default: true,
  },
  {
    name: 'Celestial - Guitar',
    url: '/inputs/Celestial - Guitar.wav',
  },
  {
    name: 'Cream - Guitar',
    url: '/inputs/Cream - Guitar.wav',
  },
  {
    name: 'Downtown - Bass',
    url: '/inputs/Downtown - Bass.wav',
  },
  {
    name: "Drivin' - Bass",
    url: "/inputs/Drivin' - Bass.wav",
  },
  {
    name: 'Fear - Guitar',
    url: '/inputs/Fear - Guitar.wav',
  },
  {
    name: 'Frogger - Bass',
    url: '/inputs/Frogger - Bass.wav',
  },
  {
    name: 'Garden - Bass',
    url: '/inputs/Garden - Bass.wav',
  },
  {
    name: 'Hammer Lead - Guitar',
    url: '/inputs/Hammer Lead - Guitar.wav',
  },
  {
    name: 'Harmonics - Guitar',
    url: '/inputs/Harmonics - Guitar.wav',
  },
  {
    name: 'Honky - Guitar',
    url: '/inputs/Honky - Guitar.wav',
  },
  {
    name: 'Hotrod - Guitar',
    url: '/inputs/Hotrod - Guitar.wav',
  },
  {
    name: 'Jazz Hop - Guitar',
    url: '/inputs/Jazz Hop - Guitar.wav',
  },
  {
    name: 'Jazz Trot - Guitar',
    url: '/inputs/Jazz Trot - Guitar.wav',
  },
  {
    name: 'John - Guitar',
    url: '/inputs/John - Guitar.wav',
  },
  {
    name: 'Lunar - Guitar', // Lunar is contributed by Aditya Mhatre
    url: '/inputs/Lunar - Guitar.wav',
  },
  {
    name: 'Mayer - Guitar',
    url: '/inputs/Mayer - Guitar.wav',
  },
  {
    name: 'Metalcore - Guitar', // Metalcore is contributed by Aditya Mhatre
    url: '/inputs/Metalcore - Guitar.wav',
  },
  {
    name: 'Pluck - Guitar',
    url: '/inputs/Pluck - Guitar.wav',
  },
  {
    name: 'Pop Punk - Guitar',
    url: '/inputs/Pop Punk - Guitar.wav',
  },
  {
    name: 'Power - Guitar',
    url: '/inputs/Power - Guitar.wav',
  },
  {
    name: 'Progression -  Guitar',
    url: '/inputs/Progression -  Guitar.wav',
  },
  {
    name: 'Raid - Guitar',
    url: '/inputs/Raid - Guitar.wav',
  },
  {
    name: "Rollin' - Bass",
    url: "/inputs/Rollin' - Bass.wav",
  },
  {
    name: 'Rotary - Guitar',
    url: '/inputs/Rotary - Guitar.wav',
  },
  {
    name: 'Slide Lead - Guitar',
    url: '/inputs/Slide Lead - Guitar.wav',
  },
  {
    name: "Smokin' - Bass",
    url: "/inputs/Smokin' - Bass.wav",
  },
  {
    name: 'Smooth - Guitar',
    url: '/inputs/Smooth - Guitar.wav',
  },
  {
    name: 'Stroke - Guitar',
    url: '/inputs/Stroke - Guitar.wav',
  },
  {
    name: 'Tomb - Guitar',
    url: '/inputs/Tomb - Guitar.wav',
  },
];

const irs = [
  {
    name: 'None',
    url: '',
    mix: 0,
    gain: 1,
  },
  {
    name: 'Mesa 412 OS',
    url: '/irs/mesa.wav',
    mix: 1,
    gain: 3,
    default: true,
  },
  {
    name: 'Celestion G12 Vintage',
    url: '/irs/celestion.wav',
    mix: 1,
    gain: 1.75,
  },
  {
    name: 'Eminence Governor',
    url: '/irs/eminence.wav',
    mix: 1,
    gain: 3,
  },
  {
    name: 'Ampeg 8x10',
    url: '/irs/ampeg.wav',
    mix: 1,
    gain: 1.25,
  },
  {
    name: 'EMT 140 Plate Reverb',
    url: '/irs/plate.wav',
    mix: 0.3,
    gain: 1,
  },
  {
    name: 'Fender Spring Reverb',
    url: '/irs/spring.wav',
    mix: 0.35,
    gain: 1,
  },
];

const models = [
  {
    name: 'Fender Deluxe Reverb',
    url: '/models/deluxe.nam',
    default: true,
  },
  {
    name: 'Marshall JCM',
    url: '/models/jcm.nam',
  },
  {
    name: 'Vox AC10',
    url: '/models/ac10.nam',
  },
  {
    name: 'Dumble Overdrive Special',
    url: '/models/dumble.nam',
  },
  {
    name: 'Ampeg V-2',
    url: '/models/ampeg.nam',
  },
  {
    name: 'Roland JC-120',
    url: '/models/jc.nam',
  },
];

const githubBaseUrl =
  'https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public';
const isDev = process.env.NODE_ENV === 'development';

export const DEFAULT_MODELS = isDev
  ? models
  : models.map(model => ({
      ...model,
      url: `${githubBaseUrl}${model.url}`,
    }));

export const DEFAULT_IRS = isDev
  ? irs
  : irs.map(ir => ({
      ...ir,
      url: ir.url ? `${githubBaseUrl}${ir.url}` : '', // Empty string if no url
    }));

export const DEFAULT_INPUTS = isDev
  ? inputs
  : inputs.map(input => ({
      ...input,
      url: `${githubBaseUrl}${input.url}`,
    }));

export const DEFAULT_AUDIO_SRC = isDev
  ? '/inputs/placeholder.wav'
  : githubBaseUrl + '/inputs/placeholder.wav';
