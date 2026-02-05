import React, { ReactNode } from 'react';
import { T3kPlayerContextProvider, useT3kPlayerContext } from './T3kPlayerContext';
import { SettingsDialog } from '../components/SettingsDialog';

function SettingsDialogRenderer() {
  const { settingsDialog, closeSettingsDialog } = useT3kPlayerContext();

  if (!settingsDialog.isOpen || !settingsDialog.selectedModel || !settingsDialog.selectedIr) {
    return null;
  }

  return (
    <div className='neural-amp-modeler' style={{ background: 'none' }}>
      <SettingsDialog
        isOpen={true}
        onSave={() => closeSettingsDialog({ saved: true })}
        onCancel={() => closeSettingsDialog({ saved: false })}
        sourceMode={settingsDialog.sourceMode}
        selectedModel={settingsDialog.selectedModel}
        selectedIr={settingsDialog.selectedIr}
        playerId={settingsDialog.playerId}
      />
    </div>
  );
}

export function T3kPlayerProvider({ children }: { children: ReactNode }) {
  return (
    <T3kPlayerContextProvider>
      {children}
      <SettingsDialogRenderer />
    </T3kPlayerContextProvider>
  );
}
