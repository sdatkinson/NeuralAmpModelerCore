import { AudioNodes } from '../types';
import { needsMediaStreamWorkaround } from './browser';

/** Clean up all live input nodes (mediaStream, source, gain, splitter, merger, meters) */
export function cleanupLiveInputNodes(nodes: AudioNodes): void {
  if (nodes.mediaStream) {
    nodes.mediaStream.getTracks().forEach(track => track.stop());
    nodes.mediaStream = null;
  }
  if (nodes.liveSourceNode) {
    nodes.liveSourceNode.disconnect();
    nodes.liveSourceNode = null;
  }
  if (nodes.liveInputGainNode) {
    nodes.liveInputGainNode.disconnect();
    nodes.liveInputGainNode = null;
  }
  if (nodes.channelSplitterNode) {
    nodes.channelSplitterNode.disconnect();
    nodes.channelSplitterNode = null;
  }
  if (nodes.channelMergerNode) {
    nodes.channelMergerNode.disconnect();
    nodes.channelMergerNode = null;
  }
  if (nodes.channel0PreviewMeter) {
    nodes.channel0PreviewMeter.disconnect();
    nodes.channel0PreviewMeter = null;
  }
  if (nodes.channel1PreviewMeter) {
    nodes.channel1PreviewMeter.disconnect();
    nodes.channel1PreviewMeter = null;
  }
}

/** Clean up output device workaround routing (Firefox/Safari) and reconnect to default destination */
export function cleanupOutputWorkaroundRouting(nodes: AudioNodes): void {
  const hadWorkaroundRouting = nodes.outputWorkaroundElement !== null || nodes.outputWorkaroundDestination !== null;

  if (nodes.outputWorkaroundElement) {
    nodes.outputWorkaroundElement.pause();
    nodes.outputWorkaroundElement.srcObject = null;
    nodes.outputWorkaroundElement = null;
  }
  if (nodes.outputWorkaroundDestination) {
    nodes.outputWorkaroundDestination.disconnect();
    nodes.outputWorkaroundDestination = null;
  }

  if (nodes.outputMeterNode && nodes.audioContext) {
    if (hadWorkaroundRouting) {
      try {
        nodes.outputMeterNode.disconnect();
      } catch (e) {
        if (!(e instanceof DOMException && e.name === 'InvalidStateError')) throw e;
      }
      nodes.outputMeterNode.connect(nodes.audioContext.destination);
    } else {
      try {
        nodes.outputMeterNode.connect(nodes.audioContext.destination);
      } catch (e) {
        if (!(e instanceof DOMException && e.name === 'InvalidAccessError')) throw e;
      }
    }
  }
}

/**
 * Tear down live input audio nodes and restore the preview signal path.
 * Used by stopLiveInput, clearLiveInputConfig, and handleLiveInputUnavailable.
 */
export function teardownLiveInput(nodes: AudioNodes, options: { muteOutput: boolean }): void {
  cleanupLiveInputNodes(nodes);

  // Reconnect file source to restore preview path
  if (nodes.sourceNode && nodes.inputGainNode) {
    try { nodes.sourceNode.disconnect(nodes.inputGainNode); } catch (e) { if (!(e instanceof DOMException && e.name === 'InvalidAccessError')) throw e; }
    nodes.sourceNode.connect(nodes.inputGainNode);
  }

  if (options.muteOutput && nodes.outputGainNode && nodes.audioContext) {
    nodes.outputGainNode.gain.setTargetAtTime(0, nodes.audioContext.currentTime, 0.01);
  }
}

/** Apply output device routing with browser-specific handling */
export async function applyOutputDeviceRouting(
  nodes: AudioNodes,
  deviceId: string | null
): Promise<void> {
  const { audioContext, outputMeterNode } = nodes;
  if (!audioContext || !outputMeterNode) return;

  if (needsMediaStreamWorkaround) {
    // Route through MediaStreamDestination + HTMLAudioElement
    if (nodes.outputWorkaroundElement) {
      nodes.outputWorkaroundElement.pause();
      nodes.outputWorkaroundElement.srcObject = null;
      nodes.outputWorkaroundElement = null;
    }
    if (nodes.outputWorkaroundDestination) {
      nodes.outputWorkaroundDestination.disconnect();
      nodes.outputWorkaroundDestination = null;
    }

    try {
      outputMeterNode.disconnect();
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'InvalidStateError')) throw e;
    }

    if (deviceId) {
      const mediaStreamDestination = audioContext.createMediaStreamDestination();
      nodes.outputWorkaroundDestination = mediaStreamDestination;
      outputMeterNode.connect(mediaStreamDestination);

      const outputElement = new Audio();
      outputElement.srcObject = mediaStreamDestination.stream;
      nodes.outputWorkaroundElement = outputElement;

      const elementWithSinkId = outputElement as HTMLAudioElement & { setSinkId?: (sinkId: string) => Promise<void> };
      if (typeof elementWithSinkId.setSinkId === 'function') {
        await elementWithSinkId.setSinkId(deviceId);
      }

      await outputElement.play();
    } else {
      outputMeterNode.connect(audioContext.destination);
    }
  } else {
    const contextWithSinkId = audioContext as AudioContext & { setSinkId?: (sinkId: string) => Promise<void> };
    if (typeof contextWithSinkId.setSinkId === 'function') {
      try {
        await contextWithSinkId.setSinkId(deviceId ?? '');
      } catch (e) {
        console.warn('[Audio] Failed to set AudioContext sink:', e);
      }
    }
  }
}
