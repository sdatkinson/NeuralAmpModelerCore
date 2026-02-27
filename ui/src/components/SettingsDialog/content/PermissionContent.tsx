import React from 'react';
import { Mic, Loader2, Lock } from 'lucide-react';
import { Button } from '../../ui/Button';
import { Alert } from '../../ui/Alert';
import { MicrophonePermissionStatus } from '../../../types';
import { getBrowserName, type BrowserName } from '../../../utils/browser';

interface PermissionContentProps {
  status: MicrophonePermissionStatus;
  errorMessage: string | null;
  onRequestPermission: () => void;
}

function BlockedInstructions({ browser }: { browser: BrowserName }) {
  switch (browser) {
    case 'safari':
      return (
        <>
          To fix this on Safari:
          <ol className='list-decimal list-inside mt-2 space-y-1'>
            <li>
              In the top menu bar, click Safari → Settings for This Website…
            </li>
            <li>Find Microphone</li>
            <li>Set to Allow</li>
            <li>Refresh the page</li>
          </ol>
          <br />
          Alternative: Safari → Settings → Websites → Microphone → set this site
          to Allow
        </>
      );
    case 'firefox':
      return (
        <>
          To fix this on Firefox:
          <ol className='list-decimal list-inside mt-2 space-y-1'>
            <li>Click the icon to the left of the website address</li>
            <li>Open Connection secure / Permissions</li>
            <li>Set Use the Microphone to Allow</li>
            <li>Refresh the page</li>
          </ol>
          <br />
          If blocked permanently: Type about:preferences#privacy → Permissions →
          Microphone → Settings… → remove this site from blocked list
        </>
      );
    case 'edge':
      return (
        <>
          To fix this on Edge:
          <ol className='list-decimal list-inside mt-2 space-y-1'>
            <li>Click the icon to the left of the website address</li>
            <li>Open Permissions for this site</li>
            <li>Set Microphone to Allow</li>
            <li>Refresh the page</li>
          </ol>
          <br />
          If blocked permanently: Type about:preferences#privacy → Permissions →
          Microphone → Settings… → remove this site from blocked list
        </>
      );
    case 'chrome':
      return (
        <>
          To fix this on Chrome:
          <ol className='list-decimal list-inside mt-2 space-y-1'>
            <li>
              Click the icon to the left of the website address (sliders / site
              controls icon)
            </li>
            <li>Find Microphone</li>
            <li>Set to Allow</li>
            <li>Refresh the page</li>
          </ol>
          <br />
          If you don't see Microphone: Chrome menu → Settings → Privacy &
          security → Site settings → Microphone → remove this site from "Not
          allowed"
        </>
      );
    default:
      return (
        <>
          To fix this, open your browser settings and allow microphone access
          for this site. Look for the site controls icon (lock or info) next to
          the address bar, then set Microphone to Allow and refresh the page.
        </>
      );
  }
}

export const PermissionContent: React.FC<PermissionContentProps> = ({
  status,
  errorMessage,
  onRequestPermission,
}) => {
  const isPending = status === 'pending';
  const isBlocked = status === 'blocked';
  const hasError = status === 'denied' || status === 'error';

  return (
    <div className='flex flex-col gap-4'>
      {/* Explanation */}
      <div className='flex flex-col gap-2'>
        <div className='text-base text-zinc-400'>
          Allow microphone access to connect your interface:
          <ol className='list-decimal list-inside mt-2 space-y-1'>
            <li>"Allow Microphone"</li>
            <li>"Allow while visiting the site"</li>
          </ol>
        </div>
      </div>

      {/* Blocked State - permanent block, needs browser settings */}
      {isBlocked && (
        <div className='flex flex-col gap-3'>
          <Alert variant='error'>
            <>
              {errorMessage ?? 'Microphone access was previously denied.'}
              <br />
              <br />
              <BlockedInstructions browser={getBrowserName()} />
            </>
          </Alert>
        </div>
      )}

      {/* Error State - can retry */}
      {hasError && (
        <Alert variant='error'>
          {(errorMessage ?? status === 'denied')
            ? 'Microphone access was previously denied. Try again and select "Allow."'
            : 'Unable to access microphone. Try again and select "Allow."'}
        </Alert>
      )}

      {/* Spacer */}
      <div className='flex h-2' />

      {/* Allow Microphone Button */}
      <Button
        variant='primary'
        size='lg'
        fullWidth
        leftIcon={
          isPending ? (
            <Loader2 size={18} className='animate-spin' />
          ) : (
            <Mic size={18} />
          )
        }
        onClick={onRequestPermission}
        disabled={isPending || isBlocked}
      >
        {isPending ? 'Asking for permission...' : 'Allow Microphone'}
      </Button>

      {/* Info about privacy */}
      <div className='flex gap-2 text-zinc-400'>
        <Lock size={18} />
        <span className='text-sm'>
          Your audio is processed locally and never leaves your device.
        </span>
      </div>
    </div>
  );
};

export default PermissionContent;
