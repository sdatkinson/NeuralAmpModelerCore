// Custom PostCSS plugin to scope Tailwind classes to the neural-amp-modeler component
module.exports = () => {
  return {
    postcssPlugin: 'postcss-scope-tailwind',
    AtRule: {
      // Handle @tailwind directives
      tailwind: atRule => {
        // We don't need to modify @tailwind directives
        // The scoping will happen at the rule level
      },
    },
    Rule: rule => {
      // Skip if this is already scoped or is a keyframe
      if (rule.parent.type === 'atrule' && rule.parent.name === 'keyframes') {
        return;
      }

      // Get the selector
      const selector = rule.selector;

      // Skip certain selectors that should not be scoped
      const skipSelectors = [
        ':root',
        'html',
        'body',
        '.neural-amp-modeler',
        '*',
        '*, ::before, ::after',
        '::before',
        '::after',
      ];

      // Check if selector should be skipped
      const shouldSkip = skipSelectors.some(skip => {
        if (skip.includes('*')) {
          return selector.includes('*');
        }
        return selector === skip || selector.startsWith(skip + ' ');
      });

      if (shouldSkip) {
        return;
      }

      // Scope the selector to .neural-amp-modeler
      if (selector.includes(',')) {
        // Handle multiple selectors
        const selectors = selector.split(',').map(s => s.trim());
        const scopedSelectors = selectors.map(s => `.neural-amp-modeler ${s}`);
        rule.selector = scopedSelectors.join(', ');
      } else {
        rule.selector = `.neural-amp-modeler ${selector}`;
      }
    },
  };
};

module.exports.postcss = true;
