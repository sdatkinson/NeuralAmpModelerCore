import React from 'react';
import { Mic, Loader2, AlertCircle } from 'lucide-react';
import { Button } from '../../ui/Button';
import { MicrophonePermissionStatus } from '../../../types';

interface LivePermissionContentProps {
  status: MicrophonePermissionStatus;
  errorMessage: string | null;
  onRequestPermission: () => void;
}

export const LivePermissionContent: React.FC<LivePermissionContentProps> = ({
  status,
  errorMessage,
  onRequestPermission,
}) => {
  const isPending = status === 'pending';
  const hasError = status === 'denied' || status === 'error';

  return (
    <div className='flex flex-col gap-4'>
      {/* Explanation */}
      <div className='flex flex-col gap-2'>
        <p className='text-sm text-zinc-300'>
          To use live input, we need access to your microphone or audio interface.
        </p>
        <p className='text-sm text-zinc-400'>
          Your audio is processed locally and never leaves your device.
        </p>
      </div>

      {/* Error State */}
      {hasError && (
        <div className='flex items-start gap-3 p-3 bg-red-950/50 border border-red-900/50 rounded-md'>
          <AlertCircle size={18} className='text-red-400 flex-shrink-0 mt-0.5' />
          <div className='flex flex-col gap-1'>
            <p className='text-sm text-red-300'>
              {status === 'denied'
                ? 'Microphone access was denied.'
                : 'Unable to access microphone.'}
            </p>
            <p className='text-xs text-red-400'>
              {errorMessage ?? 'Please check your browser settings and try again.'}
            </p>
          </div>
        </div>
      )}

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
        disabled={isPending}
      >
        {isPending
          ? 'Waiting for permission...'
          : hasError
            ? 'Try Again'
            : 'Allow Microphone Access'}
      </Button>

      {/* Pending helper text */}
      {isPending && (
        <p className='text-xs text-zinc-500 text-center'>
          Please respond to the browser permission prompt.
        </p>
      )}
    </div>
  );
};

export default LivePermissionContent;
