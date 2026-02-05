import React from 'react';
import ReactDOM from 'react-dom/client';
import { T3kPlayer } from './components/T3kPlayer';
import T3kAcordianPlayer from './components/Player/AcordianPlayer';
import { T3kPlayerProvider } from './context/T3kPlayerProvider';
import './index.css';
import { PREVIEW_MODE } from './types';
import { DEFAULT_INPUTS, DEFAULT_IRS, DEFAULT_MODELS } from './constants';

const PreviewContent: React.FC = () => {
  return (
    <div className='neural-amp-modeler' style={{ minHeight: '100vh' }}>
      <div className='p-5 flex justify-center items-center'>
        <div className='flex flex-col gap-4 max-w-[700px] w-full'>
          <T3kPlayer
            id='model-player'
            isLoading={false}
            previewMode={PREVIEW_MODE.MODEL}
            onModelChange={model => console.log('model changed', model)}
            onInputChange={input => console.log('input changed', input)}
            onIrChange={ir => console.log('ir changed', ir)}
            onPlay={({ model, input, ir }) =>
              console.log('play', { model, input, ir })
            }
          />
          <T3kAcordianPlayer
            id='acordian-player-1'
            previewMode={PREVIEW_MODE.MODEL}
            onModelChange={model => console.log('model changed', model)}
            onInputChange={input => console.log('input changed', input)}
            onIrChange={ir => console.log('ir changed', ir)}
            onPlay={({ model, input, ir }) =>
              console.log('play', { model, input, ir })
            }
            getData={async () => ({ models: DEFAULT_MODELS, irs: DEFAULT_IRS, inputs: DEFAULT_INPUTS })}
          />
          <T3kAcordianPlayer
            id='acordian-player-2'
            previewMode={PREVIEW_MODE.IR}
            onModelChange={model => console.log('model changed', model)}
            onInputChange={input => console.log('input changed', input)}
            onIrChange={ir => console.log('ir changed', ir)}
            onPlay={({ model, input, ir }) =>
              console.log('play', { model, input, ir })
            }
            getData={async () => ({ models: DEFAULT_MODELS, irs: DEFAULT_IRS, inputs: DEFAULT_INPUTS })}
          />
          <T3kAcordianPlayer
            id='acordian-player-3'
            previewMode={PREVIEW_MODE.MODEL}
            onModelChange={model => console.log('model changed', model)}
            onInputChange={input => console.log('input changed', input)}
            onIrChange={ir => console.log('ir changed', ir)}
            onPlay={({ model, input, ir }) =>
              console.log('play', { model, input, ir })
            }
            getData={async () => ({ models: DEFAULT_MODELS, irs: DEFAULT_IRS, inputs: DEFAULT_INPUTS })}
          />
          <T3kAcordianPlayer
            id='acordian-player-4'
            previewMode={PREVIEW_MODE.IR}
            onModelChange={model => console.log('model changed', model)}
            onInputChange={input => console.log('input changed', input)}
            onIrChange={ir => console.log('ir changed', ir)}
            onPlay={({ model, input, ir }) =>
              console.log('play', { model, input, ir })
            }
            getData={async () => ({ models: DEFAULT_MODELS, irs: DEFAULT_IRS, inputs: DEFAULT_INPUTS })}
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

// Add back the ReactDOM rendering code
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Preview />
  </React.StrictMode>
);

export default Preview;
