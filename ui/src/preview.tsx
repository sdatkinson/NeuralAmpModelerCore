import React from 'react';
import ReactDOM from 'react-dom/client';
import { T3kPlayer } from './components/T3kPlayer';
import { T3kPlayerContextProvider } from './context/T3kPlayerContext';
import './index.css';
import { PREVIEW_MODE } from './types';

const PreviewContent: React.FC = () => {
  return (
    <div className='p-5 flex justify-center items-center'>
      <div className='flex flex-col gap-4 max-w-[700px] w-full'>
        <T3kPlayer 
          isLoading={false}
          previewMode={PREVIEW_MODE.MODEL}
          onModelChange={(model) => console.log('model changed', model)}
          onInputChange={(input) => console.log('input changed', input)}
          onIrChange={(ir) => console.log('ir changed', ir)}
          onPlay={({ model, input, ir }) => console.log('play', { model, input, ir })}
        />
      </div>
    </div>
  );
};

const Preview: React.FC = () => {
  return (
    <T3kPlayerContextProvider>
      <PreviewContent />
    </T3kPlayerContextProvider>
  );
};

// Add back the ReactDOM rendering code
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Preview />
  </React.StrictMode>
);

export default Preview;
