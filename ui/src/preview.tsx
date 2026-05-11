import React from 'react';
import ReactDOM from 'react-dom/client';
import { T3kPlayer } from './components/T3kPlayer';
import { T3kDisabledPlayer } from './components/T3kDisabledPlayer';
import T3kAcordianPlayer from './components/Player/AcordianPlayer';
import T3kSlimPlayer from './components/Player/SlimPlayer';
import { T3kPlayerProvider } from './context/T3kPlayerProvider';
import './index.css';
import { InfoIcon } from 'lucide-react';
import { PREVIEW_MODE } from './types';
import { DEFAULT_INPUTS, DEFAULT_IRS, DEFAULT_MODELS } from './constants';

// Legacy-only set: tag every default as A1 so architecture='2' yields no matches.
const A1_MODELS = DEFAULT_MODELS.filter(m => m.architecture !== '2').map(m => ({
  ...m,
  architecture: '1' as const,
})) as unknown as typeof DEFAULT_MODELS;

// Mixed set: real A2 entries from DEFAULT_MODELS sit alongside the tagged A1 ones,
// so architecture='2' filters down to a playable subset.
const MIXED_MODELS = DEFAULT_MODELS.map(m =>
  m.architecture === '2' ? { ...m, default: true } : { ...m, architecture: '1' as const }
) as unknown as typeof DEFAULT_MODELS;

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <p className='text-xs uppercase tracking-wider text-zinc-500'>{children}</p>
);

const PreviewContent: React.FC = () => {
  return (
    <div className='neural-amp-modeler' style={{ minHeight: '100vh' }}>
      <div className='p-5 flex justify-center items-center'>
        <div className='flex flex-col gap-6 max-w-[700px] w-full'>
          <SectionLabel>SlimPlayer — no architecture filter (A1 model)</SectionLabel>
          <div className='flex items-center gap-3 p-3 rounded-lg bg-zinc-900 border border-zinc-800'>
            <T3kSlimPlayer
              id='slim-player-1'
              getData={async () => ({
                model: A1_MODELS[0],
                ir: DEFAULT_IRS[1],
                input: DEFAULT_INPUTS[0],
              })}
              onPlayDemo={({ model, input, ir }) =>
                console.log('slim play', { model, input, ir })
              }
            />
            <span className='text-sm text-zinc-400'>
              {A1_MODELS[0].name} - {DEFAULT_INPUTS[0].name} - {DEFAULT_IRS[1].name}
            </span>
          </div>

          <SectionLabel>SlimPlayer — disabled prop</SectionLabel>
          <div className='flex items-center gap-3 p-3 rounded-lg bg-zinc-900 border border-zinc-800'>
            <T3kSlimPlayer
              id='slim-player-disabled'
              disabled
              getData={async () => ({
                model: A1_MODELS[0],
                ir: DEFAULT_IRS[1],
                input: DEFAULT_INPUTS[0],
              })}
            />
            <span className='text-sm text-zinc-400'>Forced disabled</span>
          </div>

          <SectionLabel>T3kPlayer — baseline (no architecture filter)</SectionLabel>
          <T3kPlayer
            id='player-baseline'
            isLoading={false}
            previewMode={PREVIEW_MODE.MODEL}
            models={A1_MODELS}
            onPlayDemo={({ model, input, ir }) =>
              console.log('play', { model, input, ir })
            }
          />

          <SectionLabel>T3kPlayer — architecture="2", no matching models → disabled card</SectionLabel>
          <T3kPlayer
            id='player-arch-no-match'
            architecture='2'
            models={A1_MODELS}
            infoSlot={<InfoIcon size={20} className='text-zinc-400' />}
          />

          <SectionLabel>T3kPlayer — architecture="2", mixed models filtered down</SectionLabel>
          <T3kPlayer
            id='player-arch-match'
            architecture='2'
            models={MIXED_MODELS}
            onPlayDemo={({ model }) => console.log('arch-match play', model)}
          />

          <SectionLabel>T3kPlayer — disabled prop</SectionLabel>
          <T3kPlayer
            id='player-forced-disabled'
            disabled
            models={A1_MODELS}
            infoSlot={<InfoIcon size={20} className='text-zinc-400' />}
          />

          <SectionLabel>T3kDisabledPlayer — standalone export</SectionLabel>
          <T3kDisabledPlayer
            infoSlot={<InfoIcon size={20} className='text-zinc-400' />}
          />

          <SectionLabel>AcordianPlayer — baseline</SectionLabel>
          <T3kAcordianPlayer
            id='acordian-baseline'
            previewMode={PREVIEW_MODE.MODEL}
            getData={async () => ({
              models: A1_MODELS,
              irs: DEFAULT_IRS,
              inputs: DEFAULT_INPUTS,
            })}
          />

          <SectionLabel>AcordianPlayer — disabled prop</SectionLabel>
          <T3kAcordianPlayer
            id='acordian-forced-disabled'
            disabled
            previewMode={PREVIEW_MODE.MODEL}
            getData={async () => ({
              models: A1_MODELS,
              irs: DEFAULT_IRS,
              inputs: DEFAULT_INPUTS,
            })}
          />
        </div>
      </div>
    </div>
  );
};

const Preview: React.FC = () => {
  return (
    <T3kPlayerProvider>
      <PreviewContent />
    </T3kPlayerProvider>
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Preview />
  </React.StrictMode>
);

export default Preview;
