function hex(color) {
  return `#${color.hex}`;
}

function rgb(color, alpha) {
  return `rgb(${color.rgb.join(" ")} / ${alpha})`;
}

function createRosePineTheme(colors) {
  return {
    plain: {
      color: hex(colors.text),
      backgroundColor: hex(colors.base),
    },
    styles: [
      {
        types: ["comment", "prolog", "cdata"],
        style: {
          color: hex(colors.subtle),
          fontStyle: "italic",
        },
      },
      {
        types: ["punctuation"],
        style: {
          color: hex(colors.subtle),
        },
      },
      {
        types: ["namespace"],
        style: {
          opacity: 0.7,
        },
      },
      {
        types: ["constant"],
        style: {
          color: hex(colors.text),
          fontStyle: "italic",
        },
      },
      {
        types: ["delimiter", "important", "atrule", "operator", "keyword"],
        style: {
          color: hex(colors.pine),
        },
      },
      {
        types: ["tag", "variable", "regex", "class-name", "selector", "inserted"],
        style: {
          color: hex(colors.foam),
        },
      },
      {
        types: ["boolean", "entity", "number", "symbol", "function"],
        style: {
          color: hex(colors.rose),
        },
      },
      {
        types: ["string", "char", "property", "attr-value"],
        style: {
          color: hex(colors.gold),
        },
      },
      {
        types: ["parameter", "url", "name", "attr-name", "builtin"],
        style: {
          color: hex(colors.iris),
        },
      },
      {
        types: ["deleted"],
        style: {
          color: hex(colors.love),
        },
      },
      {
        types: ["inserted"],
        style: {
          background: rgb(colors.foam, 0.12),
        },
      },
      {
        types: ["deleted"],
        style: {
          background: rgb(colors.love, 0.12),
        },
      },
      {
        types: [
          "italic",
          "selector",
          "doctype",
          "attr-name",
          "inserted",
          "deleted",
          "parameter",
          "url",
        ],
        style: {
          fontStyle: "italic",
        },
      },
      {
        types: ["bold"],
        style: {
          fontWeight: "bold",
        },
      },
      {
        types: ["url"],
        style: {
          textDecoration: "underline",
        },
      },
    ],
  };
}

async function createRosePineThemes() {
  const { variantColors } = await import("@rose-pine/palette");

  return {
    rosePine: createRosePineTheme(variantColors.main),
    rosePineDawn: createRosePineTheme(variantColors.dawn),
  };
}

module.exports = { createRosePineThemes };
