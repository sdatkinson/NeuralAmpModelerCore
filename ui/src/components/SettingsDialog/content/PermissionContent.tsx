import React from 'react';
import { Mic, Loader2 } from 'lucide-react';
import { Button } from '../../ui/Button';
import { Alert } from '../../ui/Alert';
import { MicrophonePermissionStatus } from '../../../types';

type SourceMode = 'preview' | 'live';

interface PermissionContentProps {
  status: MicrophonePermissionStatus;
  errorMessage: string | null;
  onRequestPermission: () => void;
  sourceMode: SourceMode;
}

export const PermissionContent: React.FC<PermissionContentProps> = ({
  status,
  errorMessage,
  onRequestPermission,
  sourceMode,
}) => {
  const isPending = status === 'pending';
  const isBlocked = status === 'blocked';
  const hasError = status === 'denied' || status === 'error';

  const isLiveMode = sourceMode === 'live';

  return (
    <div className='flex flex-col gap-4'>
      {/* Explanation */}
      <div className='flex flex-col gap-2'>
        <p className='text-sm text-zinc-300'>
          {isLiveMode
            ? 'To use live input, we need access to your microphone or audio interface.'
            : 'To show your audio device names, we need microphone permission.'}
        </p>
        <p className='text-sm text-zinc-400'>
          Your audio is processed locally and never leaves your device.
        </p>
      </div>

      {/* Blocked State - permanent block, needs browser settings */}
      {isBlocked && (
        <div className='flex flex-col gap-3'>
          <Alert
            variant='error'
            description={errorMessage ?? 'Permission was denied in your browser settings.'}
          >
            Microphone access is blocked.
          </Alert>
          <Alert variant='info'>
            <p className='font-medium'>To enable microphone access:</p>
            <ol className='text-xs text-zinc-400 list-decimal list-inside space-y-1 mt-1'>
              <li>Click the lock or site settings icon in your browser&apos;s address bar</li>
              <li>Find &quot;Microphone&quot; in the permissions list</li>
              <li>Change the setting to &quot;Allow&quot;</li>
              <li>Refresh this page</li>
            </ol>
          </Alert>
        </div>
      )}

      {/* Error State - can retry */}
      {hasError && (
        <Alert
          variant='error'
          description={errorMessage ?? 'Please check your browser settings and try again.'}
        >
          {status === 'denied'
            ? 'Microphone access was denied.'
            : 'Unable to access microphone.'}
        </Alert>
      )}

      {/* Allow Microphone Button - hide when blocked */}
      {!isBlocked && (
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
          disabled={isPending}
        >
          {isPending
            ? 'Waiting for permission...'
            : hasError
              ? 'Try Again'
              : 'Allow Microphone Access'}
        </Button>
      )}

      {/* Pending helper text */}
      {isPending && (
        <p className='text-xs text-zinc-500 text-center'>
          Please respond to the browser permission prompt.
        </p>
      )}
    </div>
  );
};

export default PermissionContent;
