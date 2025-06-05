const FFT_SIZE = 256;

const rafIter = () => {
  let id: number;
  return {
    async next() {
      const promise = new Promise<number>((resolve) => {
        id = requestAnimationFrame(resolve);
      });
      await promise;
      return { value: undefined, done: false };
    },
    async return() {
      cancelAnimationFrame(id);
      return { value: undefined, done: true };
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  };
};

export const setupVisualizer = (
  canvas: HTMLCanvasElement,
  audioCtx: AudioContext,
  bg = '#18181b',
  useGradient = true
): AnalyserNode => {
  const canvasCtx = canvas.getContext('2d');
  if (canvasCtx === null) {
    throw new Error('canvasCtx was null');
  }

  const analyzer = audioCtx.createAnalyser();
  analyzer.fftSize = FFT_SIZE;
  const analyzeResultArray = new Uint8Array(analyzer.fftSize);

  (async () => {
    const { height, width } = canvas;
    const gap = width / analyzeResultArray.length;

    // Create linear gradient
    const gradient = canvasCtx.createLinearGradient(0, 0, width, 0);
    if (useGradient) {
      gradient.addColorStop(0, 'red');
      gradient.addColorStop(0.5, 'yellow');
      gradient.addColorStop(1, 'blue');
    } else {
      gradient.addColorStop(0, 'white');
      gradient.addColorStop(1, 'white');
    }

    for await (const _ of rafIter()) {
      analyzer.getByteTimeDomainData(analyzeResultArray);

      canvasCtx.fillStyle = bg;
      canvasCtx.fillRect(0, 0, width, height);

      canvasCtx.beginPath();
      canvasCtx.lineWidth = 2;

      for (let i = 0; i < analyzeResultArray.length; i++) {
        const data = analyzeResultArray[i];
        const x = gap * i;
        const y = height * (data / 256);

        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }
      }

      canvasCtx.strokeStyle = gradient;
      canvasCtx.lineTo(width, height / 2);
      canvasCtx.stroke();
    }
  })();

  return analyzer;
};

export const initVisualizer = ({
  canvas,
  bg = '#18181b',
  useGradient = true,
}: {
  canvas: HTMLCanvasElement | null;
  bg?: string;
  useGradient?: boolean;
}) => {
  if (!canvas) return;
  // draw a line across the canvas as default state using the gradient
  const canvasCtx = canvas.getContext('2d');
  if (canvasCtx === null) {
    throw new Error('canvasCtx was null');
  }
  const { height, width } = canvas;
  const gradient = canvasCtx.createLinearGradient(0, 0, width, 0);
  if (useGradient) {
    gradient.addColorStop(0, 'red');
    gradient.addColorStop(0.5, 'yellow');
    gradient.addColorStop(1, 'blue');
  } else {
    gradient.addColorStop(0, 'white');
    gradient.addColorStop(1, 'white');
  }
  canvasCtx.fillStyle = bg;
  canvasCtx.lineWidth = 2;
  canvasCtx.fillRect(0, 0, width, height);
  canvasCtx.beginPath();
  canvasCtx.strokeStyle = gradient;
  canvasCtx.moveTo(0, height / 2);
  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();
}; 