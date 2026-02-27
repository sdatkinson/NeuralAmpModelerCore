import React from 'react';
import { ReactNode } from 'react';

interface TabProps {
  tabs: (string | ReactNode)[];
  activeTab: number;
  setActiveTab: (index: number) => void;
  trackColor?: string;
  trackSize?: number;
  tightLines?: boolean;
}

const Tabs = ({
  tabs,
  activeTab,
  setActiveTab,
  trackColor,
  trackSize = 2,
  tightLines = false,
}: TabProps) => {
  return (
    <div className={`flex width-full ${tightLines ? 'gap-8' : 'gap-0'}`}>
      {tabs.map((tab, index) => (
        <button
          key={index}
          className={`group flex flex-1 justify-center ${tightLines ? 'px-0' : 'px-4'} py-3 font-semibold transition-colors duration-150 ${
            index === activeTab
              ? `active text-white ${trackSize === 1 ? 'border-b' : 'border-b-2'} border-white`
              : `text-zinc-400 hover:text-white ${trackSize === 1 ? 'border-b' : 'border-b-2'} ${trackColor ? `border-${trackColor}` : 'border-zinc-700'}`
          }`}
          onClick={() => setActiveTab(index)}
        >
          {tab}
        </button>
      ))}
    </div>
  );
};

export { Tabs };
