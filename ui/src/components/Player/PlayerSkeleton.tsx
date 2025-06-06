import React from 'react';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';
import { LogoSm } from '../ui/LogoSm';

export const PlayerSkeleton: React.FC = () => {
  return (
    <div className="bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full opacity-50 flex flex-col gap-6">
      <div className="flex items-center gap-4 overflow-hidden">
        <button className="p-0 focus:outline-none cursor-not-allowed" disabled>
          <Play />
        </button>
        <button className="p-0 focus:outline-none cursor-not-allowed opacity-60" disabled>
          <Skip opacity={0.6} />
        </button>
        <div className="flex text-sm font-mono gap-2 text-zinc-400">
          <span>0:00</span>
          <span> / </span>
          <span>0:00</span>
        </div>
        <div className="flex-1">
          <div
            className="w-full h-[130px] bg-zinc-900 rounded"
            style={{ marginBottom: -20, marginTop: -20 }}
          />
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <div className="flex flex-row items-center gap-4 flex-wrap">
          <div className="flex-1 min-w-[0px]">
            <div className="flex w-full flex-1 min-w-[0px]">
              <Select options={[]} label="Model" onChange={() => {}} disabled />
            </div>
          </div>
          <div className="flex items-center pt-[24px] flex-shrink-0">
            <ToggleSimple label="" onChange={() => {}} isChecked={true} disabled />
          </div>
        </div>
        <div className="flex flex-col sm:flex-row items-center gap-2 sm:gap-6">
          <div className="w-full sm:w-1/2">
            <Select options={[]} label="Input" onChange={() => {}} disabled />
          </div>
          <div className="w-full sm:w-1/2">
            <Select options={[]} label="IR" onChange={() => {}} disabled />
          </div>
        </div>
      </div>
      <a href="https://www.tone3000.com" target='_blank' className="flex flex-row gap-2 items-center self-end">
        <p className="text-zinc-400 text-xs">Powered by</p>
        <LogoSm width={42} height={14} />
      </a>
    </div>
  );
}; 