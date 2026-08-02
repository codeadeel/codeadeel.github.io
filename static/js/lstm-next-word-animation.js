(function () {
  "use strict";

  if (window.__lstmNextWordAnimationLoaded) return;
  window.__lstmNextWordAnimationLoaded = true;

  let activeFullscreenController = null;

  const B = 3;
  const T = 4;
  const E = 3;
  const H = 4;
  const V = 14;

  const PRE_STAGE_COUNT = 6;
  const CELL_OPERATION_COUNT = 8;
  const RECURRENCE_START = PRE_STAGE_COUNT;
  const METRICS_POSITION =
    RECURRENCE_START + T * CELL_OPERATION_COUNT;
  const FINAL_POSITION = METRICS_POSITION;

  const GATE_ORDER = ["i", "f", "g", "o"];
  const GATE_NAMES = {
    i: "input gate",
    f: "forget gate",
    g: "candidate",
    o: "output gate",
  };
  const GATE_ROLES = {
    i: "write permission",
    f: "old-memory retention",
    g: "signed content proposal",
    o: "hidden-state visibility",
  };
  const GATE_NOTES = {
    i: {
      raw:
        "Unbounded write score. Sigmoid turns this score into write permission between 0 and 1.",
      activated:
        "Write permission from 0 to 1: 0 blocks the candidate, while 1 accepts it fully.",
    },
    f: {
      raw:
        "Unbounded retention score. Sigmoid turns this score into an old-memory keep fraction between 0 and 1.",
      activated:
        "Old-memory retention from 0 to 1: 0 removes the previous value, while 1 carries it forward.",
    },
    g: {
      raw:
        "Unbounded proposed-content score. Tanh turns this score into signed content between −1 and 1.",
      activated:
        "Signed content proposal from −1 to 1. The input gate decides how much of this proposal is written.",
    },
    o: {
      raw:
        "Unbounded visibility score. Sigmoid turns this score into hidden-state visibility between 0 and 1.",
      activated:
        "Hidden-state visibility from 0 to 1: it scales tanh(c_t) without erasing the cell memory c_t.",
    },
  };
  const FEATURE_NAMES = ["h0", "h1", "h2", "h3"];
  const EMBEDDING_NAMES = ["e0", "e1", "e2"];

  const VOCABULARY = [
    "<PAD>",
    "<UNK>",
    "<EOS>",
    "cats",
    "chase",
    "small",
    "mice",
    "birds",
    "can",
    "fly",
    "dogs",
    "guard",
    "family",
    "homes",
  ];

  const DATASET = [
    {
      name: "S0",
      text: "cats chase small mice",
      tokens: ["cats", "chase", "small", "mice"],
      inputIDs: [3, 4, 5, 6],
      targetIDs: [4, 5, 6, 2],
      length: 4,
    },
    {
      name: "S1",
      text: "birds can fly",
      tokens: ["birds", "can", "fly"],
      inputIDs: [7, 8, 9, 0],
      targetIDs: [8, 9, 2, 0],
      length: 3,
    },
    {
      name: "S2",
      text: "dogs guard family homes",
      tokens: ["dogs", "guard", "family", "homes"],
      inputIDs: [10, 11, 12, 13],
      targetIDs: [11, 12, 13, 2],
      length: 4,
    },
  ];

  /*
   * Parameters copied from a deterministic CPU PyTorch run:
   * nn.Embedding(14, 3, padding_idx=0)
   * nn.LSTM(3, 4, batch_first=True)
   * nn.Linear(4, 14)
   *
   * PyTorch stores the four LSTM blocks in i, f, g, o order.
   */
  const MODEL = {
    embedding: [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.75534964, -1.40988576, 2.18215466],
      [0.29053447, 2.04539967, 2.37358212],
      [1.28989911, -0.63660705, 0.61670232],
      [-0.3610861, -1.20767176, 0.17158283],
      [1.85629261, 1.53975153, 0.93603975],
      [-1.04058969, 0.7931565, 1.5975374],
      [-0.2321184, -1.75388432, 0.06803165],
      [2.54113173, -1.29760051, 0.68619418],
      [0.95531577, -1.9799279, -0.12083627],
      [-0.18137781, 1.32967794, 1.86533689],
      [-0.14268459, -1.50068593, 0.73676622],
    ],
    weightIH: [
      [1.04236317, -0.75674367, 0.40166274],
      [0.76312506, -0.04932747, 1.05762494],
      [0.601493, -0.89578527, 1.08510947],
      [0.34457147, -0.31858838, 1.19313848],
      [0.89778543, 0.72128195, 0.852795],
      [-0.26045978, -1.04127657, -0.94946849],
      [-0.12904209, 0.18208773, 0.32548931],
      [-0.55188602, -0.13045976, -0.9632687],
      [-0.55075473, 1.56544816, 1.53164005],
      [-2.03597236, -0.01362013, 1.11297631],
      [-0.34229782, -0.14896078, -1.246297],
      [-1.47566152, -1.15272272, -1.02377355],
      [0.5101248, -0.36767074, 1.35035944],
      [0.90685153, -0.3028118, 1.20330477],
      [0.73314679, -3.33714056, -0.50698799],
      [0.32009175, -0.62543124, 1.67723751],
    ],
    weightHH: [
      [1.31654453, 0.49835581, -0.69666785, -0.68554986],
      [0.37945396, -0.16564806, -0.55266279, 0.03126813],
      [-0.06618641, -0.03056056, -0.03146827, 0.0075282],
      [0.1600716, -0.13077436, -0.37537822, -0.38057464],
      [-1.36783576, -0.21556814, 0.27143347, -1.43259299],
      [-1.19591236, 0.1387029, 0.12132665, -0.33404848],
      [0.35984066, 0.63721865, 0.13020051, -0.04778055],
      [0.69098848, 0.3478041, -0.39462772, 1.61669207],
      [0.22223471, 0.33699819, 0.50237346, -0.58660495],
      [0.08910973, 0.13413224, -0.55474782, 0.62304413],
      [-0.41513547, -0.82961458, -0.52455658, -0.47204253],
      [1.27215338, -0.01922586, 0.14947768, -1.75933206],
      [0.22484308, -0.18525706, -0.41295156, -0.02036952],
      [0.22866976, -0.20301279, -0.64753455, -0.09055107],
      [-0.07588243, 0.55086696, 0.10700358, 0.10548686],
      [0.39278275, -0.14182238, -0.69267213, -0.2155598],
    ],
    biasIH: [
      0.94271111, 1.20190179, 0.61417294, 1.22534585,
      0.31268859, -0.70784914, 0.33131942, -0.8074556,
      0.23614287, 0.11134267, -0.52063626, 0.38510317,
      1.46753526, 1.28358829, 0.54078877, 1.42459822,
    ],
    biasHH: [
      0.64128369, 1.05968046, 0.84923977, 1.45379663,
      0.33205101, -0.70796233, 0.36438158, -0.78688204,
      0.08097837, 0.14090136, -0.39080158, 0.74452114,
      1.4506886, 1.37819052, 0.46191984, 1.14112902,
    ],
    outputWeight: [
      [-0.03489868, 0.07391331, 0.56622046, -0.04266233],
      [-0.06338412, 0.10108411, 0.74011517, -0.0609721],
      [-3.14525533, 3.99060726, -3.27747345, 3.85310674],
      [-0.10865838, 0.12751108, 0.84434968, -0.07517134],
      [2.84417224, 3.3712759, -6.8479023, -3.34581852],
      [5.64870834, 5.54170847, 8.36971474, -6.42817879],
      [4.081779, -3.63059545, -4.17489958, 3.10928392],
      [-0.12648307, 0.11669104, 0.71737242, -0.0691683],
      [4.44975138, -4.80285549, 3.63538051, -3.70917344],
      [4.1561842, 3.12944651, 4.26294851, 4.60760021],
      [-0.05708993, 0.06288002, 0.48084021, -0.03955138],
      [-2.86223245, -3.19351006, -4.66388845, -4.11256504],
      [-3.00230622, -4.54274035, 0.47432366, 4.15271711],
      [-3.91563392, 3.98976088, 3.75705791, -3.59315705],
    ],
    outputBias: [
      -1.045452, -0.86126214, 0.53421056, -0.74532259,
      -0.59553069, -0.58966798, -0.30620095, -0.72908002,
      2.20579267, 2.39002895, -1.09233308, -0.09894389,
      0.0724387, 2.51864576,
    ],
  };

  const MACRO_STAGES = [
    { key: "text", label: "Text", position: 0 },
    { key: "tokens", label: "Tokens", position: 1 },
    { key: "ids", label: "IDs", position: 2 },
    { key: "shift", label: "Shift targets", position: 3 },
    { key: "padding", label: "Padding + mask", position: 4 },
    { key: "embedding", label: "Embedding", position: 5 },
    { key: "recurrence", label: "LSTM recurrence", position: 6 },
    { key: "metrics", label: "Masked loss", position: 38 },
  ];

  const CELL_OPERATIONS = [
    "Select x_t",
    "Input affine",
    "Hidden affine",
    "Add + split gates",
    "Activate gates",
    "Update cell c_t",
    "Update hidden h_t",
    "Logits + next word",
  ];

  function zeros(rows, columns) {
    return Array.from({ length: rows }, function () {
      return Array(columns).fill(0);
    });
  }

  function cloneMatrix(matrix) {
    return matrix.map(function (row) {
      return row.slice();
    });
  }

  function transpose(matrix) {
    return matrix[0].map(function (_value, column) {
      return matrix.map(function (row) {
        return row[column];
      });
    });
  }

  function matMul(left, right) {
    const output = zeros(left.length, right[0].length);
    for (let row = 0; row < left.length; row += 1) {
      for (let column = 0; column < right[0].length; column += 1) {
        for (let inner = 0; inner < left[0].length; inner += 1) {
          output[row][column] += left[row][inner] * right[inner][column];
        }
      }
    }
    return output;
  }

  function addFourParts(inputPart, hiddenPart, biasIH, biasHH) {
    return inputPart.map(function (row, rowIndex) {
      return row.map(function (value, column) {
        return (
          value +
          hiddenPart[rowIndex][column] +
          biasIH[column] +
          biasHH[column]
        );
      });
    });
  }

  function splitGates(matrix) {
    return {
      i: matrix.map(function (row) { return row.slice(0, H); }),
      f: matrix.map(function (row) { return row.slice(H, 2 * H); }),
      g: matrix.map(function (row) { return row.slice(2 * H, 3 * H); }),
      o: matrix.map(function (row) { return row.slice(3 * H, 4 * H); }),
    };
  }

  function mapMatrix(matrix, function_) {
    return matrix.map(function (row) {
      return row.map(function_);
    });
  }

  function multiplyElementwise(left, right) {
    return left.map(function (row, rowIndex) {
      return row.map(function (value, column) {
        return value * right[rowIndex][column];
      });
    });
  }

  function addElementwise(left, right) {
    return left.map(function (row, rowIndex) {
      return row.map(function (value, column) {
        return value + right[rowIndex][column];
      });
    });
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  function softmaxRows(matrix) {
    return matrix.map(function (row) {
      const maximum = Math.max.apply(null, row);
      const exponentials = row.map(function (value) {
        return Math.exp(value - maximum);
      });
      const denominator = exponentials.reduce(function (sum, value) {
        return sum + value;
      }, 0);
      return exponentials.map(function (value) {
        return value / denominator;
      });
    });
  }

  function argmax(row) {
    return row.reduce(function (best, value, index, values) {
      return value > values[best] ? index : best;
    }, 0);
  }

  function computeWalkthrough() {
    const paddedIDs = DATASET.map(function (item) {
      return item.inputIDs.slice();
    });
    const targetIDs = DATASET.map(function (item) {
      return item.targetIDs.slice();
    });
    const mask = targetIDs.map(function (row) {
      return row.map(function (target) {
        return target === 0 ? 0 : 1;
      });
    });
    const embedded = paddedIDs.map(function (row) {
      return row.map(function (id) {
        return MODEL.embedding[id].slice();
      });
    });

    let hidden = zeros(B, H);
    let cell = zeros(B, H);
    const records = [];

    for (let timeStep = 0; timeStep < T; timeStep += 1) {
      const xT = embedded.map(function (sequence) {
        return sequence[timeStep].slice();
      });
      const hiddenPrevious = cloneMatrix(hidden);
      const cellPrevious = cloneMatrix(cell);
      const inputAffine = matMul(xT, transpose(MODEL.weightIH));
      const hiddenAffine = matMul(
        hiddenPrevious,
        transpose(MODEL.weightHH)
      );
      const combined = addFourParts(
        inputAffine,
        hiddenAffine,
        MODEL.biasIH,
        MODEL.biasHH
      );
      const rawGates = splitGates(combined);
      const gates = {
        i: mapMatrix(rawGates.i, sigmoid),
        f: mapMatrix(rawGates.f, sigmoid),
        g: mapMatrix(rawGates.g, Math.tanh),
        o: mapMatrix(rawGates.o, sigmoid),
      };
      const forgetTerm = multiplyElementwise(gates.f, cellPrevious);
      const inputTerm = multiplyElementwise(gates.i, gates.g);
      const newCell = addElementwise(forgetTerm, inputTerm);
      const tanhCell = mapMatrix(newCell, Math.tanh);
      const newHidden = multiplyElementwise(gates.o, tanhCell);
      const logits = addElementwise(
        matMul(newHidden, transpose(MODEL.outputWeight)),
        Array.from({ length: B }, function () {
          return MODEL.outputBias.slice();
        })
      );
      const probabilities = softmaxRows(logits);
      const predictions = probabilities.map(argmax);

      records.push({
        timeStep: timeStep,
        xT: xT,
        hiddenPrevious: hiddenPrevious,
        cellPrevious: cellPrevious,
        inputAffine: inputAffine,
        hiddenAffine: hiddenAffine,
        combined: combined,
        rawGates: rawGates,
        gates: gates,
        forgetTerm: forgetTerm,
        inputTerm: inputTerm,
        newCell: newCell,
        tanhCell: tanhCell,
        newHidden: newHidden,
        logits: logits,
        probabilities: probabilities,
        predictions: predictions,
      });

      hidden = newHidden;
      cell = newCell;
    }

    const logits = DATASET.map(function (_item, sequenceIndex) {
      return records.map(function (record) {
        return record.logits[sequenceIndex].slice();
      });
    });
    const probabilities = DATASET.map(function (_item, sequenceIndex) {
      return records.map(function (record) {
        return record.probabilities[sequenceIndex].slice();
      });
    });
    const predictions = DATASET.map(function (_item, sequenceIndex) {
      return records.map(function (record) {
        return record.predictions[sequenceIndex];
      });
    });

    let lossSum = 0;
    let validCount = 0;
    let correct = 0;
    for (let sequence = 0; sequence < B; sequence += 1) {
      for (let timeStep = 0; timeStep < T; timeStep += 1) {
        if (mask[sequence][timeStep] === 0) continue;
        const target = targetIDs[sequence][timeStep];
        lossSum -= Math.log(
          Math.max(probabilities[sequence][timeStep][target], 1e-12)
        );
        validCount += 1;
        if (predictions[sequence][timeStep] === target) correct += 1;
      }
    }

    return {
      lengths: DATASET.map(function (item) { return item.length; }),
      paddedIDs: paddedIDs,
      targetIDs: targetIDs,
      mask: mask,
      embedded: embedded,
      records: records,
      logits: logits,
      probabilities: probabilities,
      predictions: predictions,
      validCount: validCount,
      correct: correct,
      maskedLoss: lossSum / validCount,
      accuracy: correct / validCount,
    };
  }

  const WALKTHROUGH = computeWalkthrough();

  function formatNumber(value) {
    const clean = Math.abs(value) < 0.0005 ? 0 : value;
    return clean.toFixed(2);
  }

  function formatPrecise(value) {
    const clean = Math.abs(value) < 0.0000005 ? 0 : value;
    return clean.toFixed(6);
  }

  function escapeHTML(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function matrixTable(matrix, options) {
    const settings = options || {};
    const rowLabels = settings.rowLabels || [];
    const columnLabels = settings.columnLabels || [];
    let maximum = 0;
    matrix.forEach(function (row) {
      row.forEach(function (value) {
        maximum = Math.max(maximum, Math.abs(value));
      });
    });
    maximum = maximum || 1;

    let html = '<div class="nextword-lstm__matrix-scroll">';
    html += '<table class="nextword-lstm__matrix">';
    if (columnLabels.length > 0) {
      html += "<thead><tr>";
      if (rowLabels.length > 0) html += "<th></th>";
      columnLabels.forEach(function (label, column) {
        html +=
          '<th class="' +
          (settings.selectedColumn === column
            ? "is-selected-column"
            : "") +
          '">' +
          escapeHTML(label) +
          "</th>";
      });
      html += "</tr></thead>";
    }
    html += "<tbody>";
    matrix.forEach(function (row, rowIndex) {
      html +=
        '<tr class="' +
        (settings.selectedRow === rowIndex ? "is-selected" : "") +
        '">';
      if (rowLabels.length > 0) {
        html += "<th>" + escapeHTML(rowLabels[rowIndex] || rowIndex) + "</th>";
      }
      row.forEach(function (value, column) {
        const alpha = 0.05 + 0.2 * (Math.abs(value) / maximum);
        const color =
          value >= 0
            ? "rgba(37,99,235," + alpha.toFixed(3) + ")"
            : "rgba(194,59,59," + alpha.toFixed(3) + ")";
        html +=
          '<td class="' +
          (settings.selectedColumn === column
            ? "is-selected-column"
            : "") +
          '" style="background:' +
          color +
          '">' +
          formatNumber(value) +
          "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table></div>";
    return html;
  }

  function matrixCard(title, shape, matrix, note, active, options) {
    return (
      '<article class="nextword-lstm__matrix-card ' +
      (active ? "nextword-lstm__matrix-card--active" : "") +
      (options && options.cardClass ? " " + options.cardClass : "") +
      '">' +
      '<div class="nextword-lstm__matrix-title">' +
      escapeHTML(title) +
      " · " +
      escapeHTML(shape) +
      "</div>" +
      matrixTable(matrix, options) +
      (note
        ? '<div class="nextword-lstm__matrix-note">' + note + "</div>"
        : "") +
      "</article>"
    );
  }

  function informationCard(title, shape, body, active) {
    return (
      '<article class="nextword-lstm__matrix-card ' +
      (active ? "nextword-lstm__matrix-card--active" : "") +
      '">' +
      '<div class="nextword-lstm__matrix-title">' +
      escapeHTML(title) +
      (shape ? " · " + escapeHTML(shape) : "") +
      "</div>" +
      body +
      "</article>"
    );
  }

  function dataList(items) {
    return (
      '<div class="nextword-lstm__data-list">' +
      items
        .map(function (item) {
          return (
            "<div><strong>" +
            escapeHTML(item.label) +
            "</strong><code>" +
            escapeHTML(item.value) +
            "</code></div>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function rowLabels() {
    return DATASET.map(function (item) { return item.name; });
  }

  function positionInfo(position) {
    const pre = [
      {
        kind: "pre",
        key: "text",
        title: "Raw training text",
        stage: 0,
      },
      {
        kind: "pre",
        key: "tokens",
        title: "Split text into tokens",
        stage: 1,
      },
      {
        kind: "pre",
        key: "ids",
        title: "Replace tokens with vocabulary IDs",
        stage: 2,
      },
      {
        kind: "pre",
        key: "shift",
        title: "Shift inputs and next-token targets",
        stage: 3,
      },
      {
        kind: "pre",
        key: "padding",
        title: "Pad the batch and build the loss mask",
        stage: 4,
      },
      {
        kind: "pre",
        key: "embedding",
        title: "Look up embedding vectors",
        stage: 5,
      },
    ];
    if (position < RECURRENCE_START) return pre[position];
    if (position < METRICS_POSITION) {
      const offset = position - RECURRENCE_START;
      return {
        kind: "lstm",
        key: "recurrence",
        title: CELL_OPERATIONS[offset % CELL_OPERATION_COUNT],
        stage: 6,
        timeStep: Math.floor(offset / CELL_OPERATION_COUNT),
        operation: offset % CELL_OPERATION_COUNT,
      };
    }
    return {
      kind: "metrics",
      key: "metrics",
      title: "Masked next-token loss and accuracy",
      stage: 7,
    };
  }

  function renderStageNavigation(position) {
    return MACRO_STAGES.map(function (stage, index) {
      const nextPosition =
        index + 1 < MACRO_STAGES.length
          ? MACRO_STAGES[index + 1].position
          : FINAL_POSITION + 1;
      const current =
        position >= stage.position && position < nextPosition;
      const complete = position >= nextPosition;
      const button =
        '<button type="button" class="' +
        (current ? "is-current " : "") +
        (complete ? "is-complete" : "") +
        '" data-nextword-lstm-jump="' +
        stage.position +
        '"' +
        (current ? ' aria-current="step"' : "") +
        ">" +
        '<span class="nextword-lstm__stage-number">' +
        (index + 1) +
        "</span>" +
        escapeHTML(stage.label) +
        "</button>";
      const connector =
        index < MACRO_STAGES.length - 1
          ? '<span class="nextword-lstm__stage-connector ' +
            (complete ? "is-complete" : "") +
            '" aria-hidden="true"><span></span></span>'
          : "";
      return button + connector;
    }).join("");
  }

  function renderTimestepNavigation(info) {
    return Array.from({ length: T }, function (_value, timeStep) {
      const current = info.kind === "lstm" && info.timeStep === timeStep;
      return (
        '<button type="button" class="' +
        (current ? "is-current" : "") +
        '" data-nextword-lstm-jump="' +
        (RECURRENCE_START + timeStep * CELL_OPERATION_COUNT) +
        '">t' +
        timeStep +
        "</button>"
      );
    }).join("");
  }

  function renderOperationNavigation(info) {
    const timeStep = info.kind === "lstm" ? info.timeStep : 0;
    return CELL_OPERATIONS.map(function (label, operation) {
      const current =
        info.kind === "lstm" && info.operation === operation;
      return (
        '<button type="button" class="' +
        (current ? "is-current" : "") +
        '" data-nextword-lstm-jump="' +
        (RECURRENCE_START +
          timeStep * CELL_OPERATION_COUNT +
          operation) +
        '">' +
        escapeHTML(label) +
        "</button>"
      );
    }).join("");
  }

  function renderSentencePicker(selectedSentence) {
    return DATASET.map(function (item, index) {
      return (
        '<button type="button" class="' +
        (selectedSentence === index ? "is-current" : "") +
        '" data-nextword-lstm-select-sentence="' +
        index +
        '" aria-pressed="' +
        String(selectedSentence === index) +
        '">' +
        item.name +
        " · " +
        item.length +
        " words</button>"
      );
    }).join("");
  }

  function renderFeaturePicker(selectedFeature) {
    return FEATURE_NAMES.map(function (name, index) {
      return (
        '<button type="button" class="' +
        (selectedFeature === index ? "is-current" : "") +
        '" data-nextword-lstm-select-feature="' +
        index +
        '" aria-pressed="' +
        String(selectedFeature === index) +
        '">' +
        name +
        "</button>"
      );
    }).join("");
  }

  function renderGatePicker(selectedGate) {
    return GATE_ORDER.map(function (gate) {
      return (
        '<button type="button" class="nextword-lstm__gate-button nextword-lstm__gate-button--' +
        gate +
        " " +
        (selectedGate === gate ? "is-current" : "") +
        '" data-nextword-lstm-select-gate="' +
        gate +
        '" aria-pressed="' +
        String(selectedGate === gate) +
        '">' +
        gate +
        " · " +
        GATE_NAMES[gate] +
        "</button>"
      );
    }).join("");
  }

  function renderBatch(info, selectedSentence) {
    const activeTime = info.kind === "lstm" ? info.timeStep : -1;
    const revealIDs =
      info.kind !== "pre" || ["ids", "shift", "padding", "embedding"].includes(info.key);
    const revealTargets =
      info.kind !== "pre" || ["shift", "padding", "embedding"].includes(info.key);
    const revealPadding =
      info.kind !== "pre" || ["padding", "embedding"].includes(info.key);

    return DATASET.map(function (item, sequenceIndex) {
      let status = "waiting for recurrence";
      if (info.kind === "lstm") {
        const target = VOCABULARY[item.targetIDs[activeTime]];
        status =
          "t" +
          activeTime +
          " · target " +
          target +
          (item.targetIDs[activeTime] === 0 ? " · loss ignored" : "");
      } else if (info.kind === "metrics") {
        status = item.length + " scored positions";
      }
      const completed =
        info.kind === "metrics"
          ? T
          : info.kind === "lstm"
            ? activeTime + (info.operation === 7 ? 1 : 0)
            : 0;

      const tokens = item.inputIDs.map(function (id, timeStep) {
        const current = activeTime === timeStep;
        const read =
          info.kind === "metrics" ||
          (info.kind === "lstm" &&
            (timeStep < activeTime ||
              (timeStep === activeTime && info.operation === 7)));
        const padding = id === 0;
        const emptySlot = padding && !revealPadding;
        const targetID = item.targetIDs[timeStep];
        const targetText =
          targetID === 0 ? "<PAD> ignored" : VOCABULARY[targetID];
        let state = "waiting";
        if (padding) state = revealPadding ? "PAD input" : "empty slot";
        if (read) state = "calculated";
        if (current) {
          state =
            info.operation === 7
              ? "predicted"
              : padding
                ? "PAD still runs"
                : "current";
        }
        return (
          '<div class="nextword-lstm__token ' +
          (current ? "is-current-token " : "") +
          (read ? "is-read " : "") +
          (padding && revealPadding ? "is-padding " : "") +
          (emptySlot ? "is-empty-slot" : "") +
          '">' +
          '<span class="nextword-lstm__token-position">t' +
          timeStep +
          "</span>" +
          '<span class="nextword-lstm__token-word">' +
          escapeHTML(emptySlot ? "—" : VOCABULARY[id]) +
          "</span>" +
          (revealIDs && !emptySlot
            ? '<span class="nextword-lstm__token-id">id ' + id + "</span>"
            : "") +
          (revealTargets && (!emptySlot || targetID !== 0)
            ? '<span class="nextword-lstm__token-target">→ ' +
              escapeHTML(targetText) +
              "</span>"
            : "") +
          '<span class="nextword-lstm__token-state">' +
          escapeHTML(state) +
          "</span>" +
          "</div>"
        );
      }).join("");

      return (
        '<button type="button" class="nextword-lstm__batch-row ' +
        (selectedSentence === sequenceIndex ? "is-selected" : "") +
        '" data-nextword-lstm-select-sentence="' +
        sequenceIndex +
        '" aria-pressed="' +
        String(selectedSentence === sequenceIndex) +
        '">' +
        '<span class="nextword-lstm__batch-meta"><strong>' +
        item.name +
        "</strong><span>length " +
        item.length +
        "</span><span>" +
        item.length +
        " valid targets</span></span>" +
        '<span class="nextword-lstm__batch-content">' +
        '<span class="nextword-lstm__batch-text-line"><span class="nextword-lstm__batch-text">“' +
        escapeHTML(item.text) +
        '”</span><span class="nextword-lstm__lane-status">' +
        escapeHTML(status) +
        "</span></span>" +
        '<span class="nextword-lstm__token-strip">' +
        tokens +
        "</span>" +
        '<span class="nextword-lstm__lane-progress" aria-hidden="true"><span style="width:' +
        ((completed / T) * 100).toFixed(0) +
        '%"></span></span>' +
        "</span></button>"
      );
    }).join("");
  }

  function selectedTime(info) {
    return info.kind === "lstm" ? info.timeStep : 0;
  }

  function renderPreMatrices(info, selectedSentence) {
    const selected = DATASET[selectedSentence];
    if (info.key === "text") {
      return informationCard(
        "Three short training sequences",
        "3 strings",
        dataList(DATASET.map(function (item) {
          return { label: item.name, value: "“" + item.text + "”" };
        })),
        true
      );
    }
    if (info.key === "tokens") {
      return informationCard(
        "Token lists",
        "variable lengths",
        dataList(DATASET.map(function (item) {
          return {
            label: item.name,
            value: "[" + item.tokens.join(", ") + "]",
          };
        })),
        true
      );
    }
    if (info.key === "ids") {
      const vocabulary =
        '<div class="nextword-lstm__vocabulary">' +
        VOCABULARY.map(function (word, id) {
          return "<code>" + id + " → " + escapeHTML(word) + "</code>";
        }).join("") +
        "</div>";
      return (
        informationCard(
          "Vocabulary lookup",
          "V=14",
          vocabulary,
          true
        ) +
        informationCard(
          "Sequence IDs",
          "variable lengths",
          dataList(DATASET.map(function (item) {
            return {
              label: item.name,
              value: "[" +
                item.inputIDs.slice(0, item.length).join(", ") +
                "]",
            };
          })),
          false
        )
      );
    }
    if (info.key === "shift") {
      return (
        informationCard(
          "Input tokens",
          "variable lengths",
          dataList(DATASET.map(function (item) {
            return {
              label: item.name,
              value: item.inputIDs
                .slice(0, item.length)
                .map(function (id) { return VOCABULARY[id]; })
                .join("  "),
            };
          })),
          true
        ) +
        informationCard(
          "Targets one position ahead",
          "variable lengths",
          dataList(DATASET.map(function (item) {
            return {
              label: item.name,
              value: item.targetIDs
                .slice(0, item.length)
                .map(function (id) { return VOCABULARY[id]; })
                .join("  "),
            };
          })),
          true
        )
      );
    }
    if (info.key === "padding") {
      return (
        matrixCard(
          "Padded input IDs",
          "(3,4)",
          WALKTHROUGH.paddedIDs,
          "S1 receives a PAD input at t3 so every row has four positions.",
          true,
          {
            rowLabels: rowLabels(),
            columnLabels: ["t0", "t1", "t2", "t3"],
            selectedRow: selectedSentence,
          }
        ) +
        matrixCard(
          "Shifted target IDs",
          "(3,4)",
          WALKTHROUGH.targetIDs,
          "Target 0 is PAD and is excluded from cross-entropy.",
          false,
          {
            rowLabels: rowLabels(),
            columnLabels: ["t0", "t1", "t2", "t3"],
            selectedRow: selectedSentence,
          }
        ) +
        matrixCard(
          "Loss mask",
          "(3,4)",
          WALKTHROUGH.mask,
          "Eleven positions have mask 1. The single padded target has mask 0.",
          false,
          {
            rowLabels: rowLabels(),
            columnLabels: ["t0", "t1", "t2", "t3"],
            selectedRow: selectedSentence,
          }
        )
      );
    }

    return (
      matrixCard(
        selected.name + " embedding sequence",
        "(4,3)",
        WALKTHROUGH.embedded[selectedSentence],
        "Each token ID selects one learned row from the embedding table.",
        true,
        {
          rowLabels: ["t0", "t1", "t2", "t3"],
          columnLabels: EMBEDDING_NAMES,
        }
      ) +
      matrixCard(
        "Embedding table",
        "(14,3)",
        MODEL.embedding,
        "The PAD row is exactly zero because padding_idx=0.",
        false,
        {
          rowLabels: VOCABULARY,
          columnLabels: EMBEDDING_NAMES,
          selectedRow: selected.inputIDs[0],
        }
      )
    );
  }

  function gateColumnLabels() {
    return GATE_ORDER.flatMap(function (gate) {
      return FEATURE_NAMES.map(function (feature) {
        return gate + ":" + feature;
      });
    });
  }

  function gateCard(gate, matrix, titlePrefix, active, selectedSentence, selectedFeature) {
    const note =
      titlePrefix === "Raw"
        ? GATE_NOTES[gate].raw
        : GATE_NOTES[gate].activated;
    return matrixCard(
      titlePrefix + " " + gate + " · " + GATE_NAMES[gate],
      "(3,4)",
      matrix,
      note,
      active,
      {
        rowLabels: rowLabels(),
        columnLabels: FEATURE_NAMES,
        selectedRow: selectedSentence,
        selectedColumn: selectedFeature,
        cardClass: "nextword-lstm__matrix-card--gate-" + gate,
      }
    );
  }

  function activatedGateEffect(gate, value) {
    if (gate === "i") {
      return (
        "The selected value <code>i=" +
        formatPrecise(value) +
        "</code> passes " +
        (value * 100).toFixed(1) +
        "% of the candidate into <code>i_t ⊙ g_t</code>."
      );
    }
    if (gate === "f") {
      return (
        "The selected value <code>f=" +
        formatPrecise(value) +
        "</code> retains " +
        (value * 100).toFixed(1) +
        "% of the previous cell value in <code>f_t ⊙ c_(t−1)</code>."
      );
    }
    if (gate === "g") {
      return (
        "The selected value <code>g=" +
        formatPrecise(value) +
        "</code> is a signed proposal. It becomes stored content only after multiplication by <code>i_t</code>."
      );
    }
    return (
      "The selected value <code>o=" +
      formatPrecise(value) +
      "</code> exposes " +
      (value * 100).toFixed(1) +
      "% of <code>tanh(c_t)</code> as <code>h_t</code>; <code>c_t</code> itself remains intact."
    );
  }

  function topPredictions(probabilities, count) {
    return probabilities
      .map(function (probability, id) {
        return { id: id, probability: probability };
      })
      .sort(function (left, right) {
        return right.probability - left.probability;
      })
      .slice(0, count);
  }

  function predictionList(record, selectedSentence) {
    const target = DATASET[selectedSentence].targetIDs[record.timeStep];
    const top = topPredictions(record.probabilities[selectedSentence], 5);
    return (
      '<div class="nextword-lstm__prediction-list">' +
      top.map(function (item, index) {
        return (
          '<div class="' +
          (item.id === target ? "is-target" : "") +
          '"><span>' +
          (index + 1) +
          ". " +
          escapeHTML(VOCABULARY[item.id]) +
          "</span><strong>" +
          (item.probability * 100).toFixed(2) +
          "%</strong></div>"
        );
      }).join("") +
      "</div>"
    );
  }

  function renderLSTMMatrices(info, selectedSentence, selectedFeature, selectedGate) {
    const record = WALKTHROUGH.records[info.timeStep];
    const common = {
      rowLabels: rowLabels(),
      selectedRow: selectedSentence,
    };
    const featureOptions = {
      rowLabels: rowLabels(),
      columnLabels: FEATURE_NAMES,
      selectedRow: selectedSentence,
      selectedColumn: selectedFeature,
    };
    if (info.operation === 0) {
      return (
        matrixCard(
          "Current embeddings x_t",
          "(3,3)",
          record.xT,
          "One embedding vector from each batch row.",
          true,
          {
            rowLabels: rowLabels(),
            columnLabels: EMBEDDING_NAMES,
            selectedRow: selectedSentence,
          }
        ) +
        matrixCard(
          "Previous hidden state h_(t−1)",
          "(3,4)",
          record.hiddenPrevious,
          "Short exposed state passed from the previous timestep.",
          false,
          featureOptions
        ) +
        matrixCard(
          "Previous cell memory c_(t−1)",
          "(3,4)",
          record.cellPrevious,
          "Longer memory path updated by forget and input gates.",
          false,
          featureOptions
        )
      );
    }
    if (info.operation === 1) {
      return (
        matrixCard(
          "Input contribution x_t W_ihᵀ",
          "(3,16)",
          record.inputAffine,
          "Sixteen outputs are four consecutive H=4 gate blocks.",
          true,
          {
            rowLabels: rowLabels(),
            columnLabels: gateColumnLabels(),
            selectedRow: selectedSentence,
            selectedColumn:
              GATE_ORDER.indexOf(selectedGate) * H + selectedFeature,
          }
        ) +
        informationCard(
          "PyTorch input weights",
          "W_ih=(16,3)",
          '<div class="nextword-lstm__formula">x_t (3,3) @ W_ihᵀ (3,16) = (3,16)</div>',
          false
        )
      );
    }
    if (info.operation === 2) {
      return (
        matrixCard(
          "Hidden contribution h_(t−1) W_hhᵀ",
          "(3,16)",
          record.hiddenAffine,
          "The previous hidden state contributes to every gate.",
          true,
          {
            rowLabels: rowLabels(),
            columnLabels: gateColumnLabels(),
            selectedRow: selectedSentence,
            selectedColumn:
              GATE_ORDER.indexOf(selectedGate) * H + selectedFeature,
          }
        ) +
        matrixCard(
          "Previous hidden state",
          "(3,4)",
          record.hiddenPrevious,
          "At t0 this matrix is all zeros.",
          false,
          featureOptions
        )
      );
    }
    if (info.operation === 3) {
      return (
        matrixCard(
          "Combined affine result",
          "(3,16)",
          record.combined,
          "input + hidden + b_ih + b_hh, then split in i, f, g, o order.",
          true,
          {
            rowLabels: rowLabels(),
            columnLabels: gateColumnLabels(),
            selectedRow: selectedSentence,
            selectedColumn:
              GATE_ORDER.indexOf(selectedGate) * H + selectedFeature,
          }
        ) +
        GATE_ORDER.map(function (gate) {
          return gateCard(
            gate,
            record.rawGates[gate],
            "Raw",
            gate === selectedGate,
            selectedSentence,
            selectedFeature
          );
        }).join("")
      );
    }
    if (info.operation === 4) {
      return GATE_ORDER.map(function (gate) {
        return gateCard(
          gate,
          record.gates[gate],
          "Activated",
          gate === selectedGate,
          selectedSentence,
          selectedFeature
        );
      }).join("");
    }
    if (info.operation === 5) {
      return (
        matrixCard(
          "Kept old memory f_t ⊙ c_(t−1)",
          "(3,4)",
          record.forgetTerm,
          "The forget gate is a retention control: 0 removes an old value and 1 keeps it.",
          false,
          featureOptions
        ) +
        matrixCard(
          "Accepted candidate i_t ⊙ g_t",
          "(3,4)",
          record.inputTerm,
          "The candidate proposes signed content; the input gate controls how much gets written.",
          false,
          featureOptions
        ) +
        matrixCard(
          "New cell memory c_t",
          "(3,4)",
          record.newCell,
          "New memory equals retained old memory plus accepted new content.",
          true,
          featureOptions
        )
      );
    }
    if (info.operation === 6) {
      return (
        matrixCard(
          "Cell memory c_t",
          "(3,4)",
          record.newCell,
          "Internal memory continues to the next cell and is not erased by the output gate.",
          false,
          featureOptions
        ) +
        matrixCard(
          "Squashed memory tanh(c_t)",
          "(3,4)",
          record.tanhCell,
          "Tanh bounds the memory view between −1 and 1 before exposure.",
          false,
          featureOptions
        ) +
        matrixCard(
          "Output gate o_t",
          "(3,4)",
          record.gates.o,
          "Visibility control from 0 to 1: it scales tanh(c_t), not c_t itself.",
          false,
          featureOptions
        ) +
        matrixCard(
          "New hidden state h_t",
          "(3,4)",
          record.newHidden,
          "The visible result h_t goes to the vocabulary head and the next timestep.",
          true,
          featureOptions
        )
      );
    }
    return (
      matrixCard(
        "Vocabulary logits",
        "(3,14)",
        record.logits,
        "One unnormalized score for every vocabulary entry.",
        true,
        {
          rowLabels: rowLabels(),
          columnLabels: VOCABULARY,
          selectedRow: selectedSentence,
        }
      ) +
      informationCard(
        DATASET[selectedSentence].name + " top next-token probabilities",
        "top 5 of V=14",
        predictionList(record, selectedSentence),
        true
      ) +
      informationCard(
        "Current batch predictions",
        "(3,)",
        dataList(DATASET.map(function (item, sequenceIndex) {
          const prediction = record.predictions[sequenceIndex];
          const target = item.targetIDs[info.timeStep];
          return {
            label: item.name,
            value:
              VOCABULARY[prediction] +
              "  | target " +
              VOCABULARY[target] +
              (target === 0 ? "  | ignored" : ""),
          };
        })),
        false
      )
    );
  }

  function renderCellMap(info, selectedGate) {
    if (info.kind !== "lstm") {
      return (
        '<div class="nextword-lstm__cell-map-summary">' +
        "<span>embedding</span><b>→</b><span>four gates</span><b>→</b>" +
        "<span>cell memory c</span><b>→</b><span>hidden state h</span><b>→</b>" +
        "<span>vocabulary logits</span></div>"
      );
    }
    const active = info.operation;
    return (
      '<div class="nextword-lstm__cell-diagram" aria-label="LSTM cell flow at timestep ' +
      info.timeStep +
      '">' +
      '<div class="nextword-lstm__cell-node ' +
      (active <= 2 ? "is-active" : "") +
      '"><small>inputs</small><strong>x_t + h_(t−1)</strong></div>' +
      '<div class="nextword-lstm__cell-arrow" aria-hidden="true">→</div>' +
      '<div class="nextword-lstm__gate-bank ' +
      (active === 3 || active === 4 ? "is-active" : "") +
      '">' +
      GATE_ORDER.map(function (gate) {
        return (
          '<span class="nextword-lstm__gate nextword-lstm__gate--' +
          gate +
          " " +
          (selectedGate === gate ? "is-selected" : "") +
          '"><b>' +
          gate +
          "</b><small>" +
          escapeHTML(GATE_NAMES[gate]) +
          "</small></span>"
        );
      }).join("") +
      "</div>" +
      '<div class="nextword-lstm__cell-arrow" aria-hidden="true">→</div>' +
      '<div class="nextword-lstm__cell-node nextword-lstm__cell-node--memory ' +
      (active === 5 ? "is-active" : "") +
      '"><small>cell memory</small><strong>c_t</strong></div>' +
      '<div class="nextword-lstm__cell-arrow" aria-hidden="true">→</div>' +
      '<div class="nextword-lstm__cell-node nextword-lstm__cell-node--hidden ' +
      (active === 6 ? "is-active" : "") +
      '"><small>hidden state</small><strong>h_t</strong></div>' +
      '<div class="nextword-lstm__cell-arrow" aria-hidden="true">→</div>' +
      '<div class="nextword-lstm__cell-node nextword-lstm__cell-node--output ' +
      (active === 7 ? "is-active" : "") +
      '"><small>prediction</small><strong>14 logits</strong></div>' +
      "</div>"
    );
  }

  function calculationRow(label, value, active) {
    return (
      "<dt" +
      (active ? ' class="is-active"' : "") +
      ">" +
      escapeHTML(label) +
      "</dt><dd" +
      (active ? ' class="is-active"' : "") +
      "><code>" +
      escapeHTML(value) +
      "</code></dd>"
    );
  }

  function dotExpression(vector, weights) {
    return vector
      .map(function (value, index) {
        return formatNumber(value) + "×" + formatNumber(weights[index]);
      })
      .join(" + ");
  }

  function scalarInspector(info, selectedSentence, selectedFeature, selectedGate) {
    if (info.kind !== "lstm") {
      return (
        '<div class="nextword-lstm__inspector-title">Selected trace</div>' +
        '<p>The exact scalar arithmetic appears after recurrence begins.</p>'
      );
    }
    const record = WALKTHROUGH.records[info.timeStep];
    const gateIndex = GATE_ORDER.indexOf(selectedGate);
    const combinedIndex = gateIndex * H + selectedFeature;
    const inputVector = record.xT[selectedSentence];
    const hiddenVector = record.hiddenPrevious[selectedSentence];
    const inputValue = record.inputAffine[selectedSentence][combinedIndex];
    const hiddenValue = record.hiddenAffine[selectedSentence][combinedIndex];
    const rawValue = record.rawGates[selectedGate][selectedSentence][selectedFeature];
    const gateValue = record.gates[selectedGate][selectedSentence][selectedFeature];
    const sequence = DATASET[selectedSentence];
    const target = sequence.targetIDs[info.timeStep];
    let rows = "";
    let traceSubject = "";

    if (info.operation === 0) {
      traceSubject = "input and previous states";
      rows += calculationRow(
        "Current token",
        VOCABULARY[sequence.inputIDs[info.timeStep]] +
          " → [" +
          inputVector.map(formatNumber).join(", ") +
          "]",
        true
      );
      rows += calculationRow(
        "Previous h",
        "[" + hiddenVector.map(formatNumber).join(", ") + "]",
        false
      );
      rows += calculationRow(
        "Previous c",
        "[" +
          record.cellPrevious[selectedSentence].map(formatNumber).join(", ") +
          "]",
        false
      );
    } else if (info.operation === 1) {
      traceSubject =
        selectedGate + " gate · " + FEATURE_NAMES[selectedFeature];
      rows += calculationRow(
        "Input dot product",
        dotExpression(inputVector, MODEL.weightIH[combinedIndex]),
        true
      );
      rows += calculationRow("Result", formatPrecise(inputValue), true);
    } else if (info.operation === 2) {
      traceSubject =
        selectedGate + " gate · " + FEATURE_NAMES[selectedFeature];
      rows += calculationRow(
        "Hidden dot product",
        dotExpression(hiddenVector, MODEL.weightHH[combinedIndex]),
        true
      );
      rows += calculationRow("Result", formatPrecise(hiddenValue), true);
    } else if (info.operation === 3) {
      traceSubject =
        selectedGate + " gate · " + FEATURE_NAMES[selectedFeature];
      rows += calculationRow("Input part", formatPrecise(inputValue), false);
      rows += calculationRow("Hidden part", formatPrecise(hiddenValue), false);
      rows += calculationRow(
        "Two biases",
        formatPrecise(MODEL.biasIH[combinedIndex]) +
          " + " +
          formatPrecise(MODEL.biasHH[combinedIndex]),
        false
      );
      rows += calculationRow(
        "Raw " + selectedGate,
        formatPrecise(inputValue) +
          " + " +
          formatPrecise(hiddenValue) +
          " + " +
          formatPrecise(MODEL.biasIH[combinedIndex]) +
          " + " +
          formatPrecise(MODEL.biasHH[combinedIndex]) +
          " = " +
          formatPrecise(rawValue),
        true
      );
    } else if (info.operation === 4) {
      traceSubject =
        selectedGate +
        " · " +
        GATE_ROLES[selectedGate] +
        " · " +
        FEATURE_NAMES[selectedFeature];
      rows += calculationRow(
        "Raw " + selectedGate,
        formatPrecise(rawValue),
        false
      );
      rows += calculationRow(
        selectedGate === "g"
          ? "Signed proposal g = tanh(raw)"
          : GATE_ROLES[selectedGate] +
            " " +
            selectedGate +
            " = sigmoid(raw)",
        formatPrecise(gateValue),
        true
      );
    } else if (info.operation === 5) {
      traceSubject =
        "cell update · " + FEATURE_NAMES[selectedFeature];
      const oldCell =
        record.cellPrevious[selectedSentence][selectedFeature];
      const f = record.gates.f[selectedSentence][selectedFeature];
      const i = record.gates.i[selectedSentence][selectedFeature];
      const g = record.gates.g[selectedSentence][selectedFeature];
      const newCell = record.newCell[selectedSentence][selectedFeature];
      rows += calculationRow(
        "Keep old",
        formatNumber(f) +
          " × " +
          formatNumber(oldCell) +
          " = " +
          formatPrecise(f * oldCell),
        false
      );
      rows += calculationRow(
        "Write new",
        formatNumber(i) +
          " × " +
          formatNumber(g) +
          " = " +
          formatPrecise(i * g),
        false
      );
      rows += calculationRow(
        "New c",
        formatPrecise(f * oldCell) +
          " + " +
          formatPrecise(i * g) +
          " = " +
          formatPrecise(newCell),
        true
      );
    } else if (info.operation === 6) {
      traceSubject =
        "hidden update · " + FEATURE_NAMES[selectedFeature];
      const newCell = record.newCell[selectedSentence][selectedFeature];
      const o = record.gates.o[selectedSentence][selectedFeature];
      const newHidden = record.newHidden[selectedSentence][selectedFeature];
      rows += calculationRow("Cell c", formatPrecise(newCell), false);
      rows += calculationRow("Output gate o", formatPrecise(o), false);
      rows += calculationRow(
        "New h",
        formatNumber(o) +
          " × tanh(" +
          formatNumber(newCell) +
          ") = " +
          formatPrecise(newHidden),
        true
      );
    } else {
      const prediction = record.predictions[selectedSentence];
      const hiddenNow = record.newHidden[selectedSentence];
      const outputWeights = MODEL.outputWeight[prediction];
      const outputDot = hiddenNow.reduce(function (sum, value, index) {
        return sum + value * outputWeights[index];
      }, 0);
      const predictionLogit = record.logits[selectedSentence][prediction];
      const probability =
        record.probabilities[selectedSentence][prediction];
      traceSubject =
        "output head · " + VOCABULARY[prediction];
      rows += calculationRow(
        "Target",
        VOCABULARY[target] + (target === 0 ? " · ignored" : ""),
        false
      );
      rows += calculationRow(
        "Hidden h_t",
        "[" + hiddenNow.map(formatNumber).join(", ") + "]",
        false
      );
      rows += calculationRow(
        "Output dot product",
        dotExpression(hiddenNow, outputWeights),
        false
      );
      rows += calculationRow(
        "Dot-product result",
        formatPrecise(outputDot),
        false
      );
      rows += calculationRow(
        "Output bias",
        formatPrecise(MODEL.outputBias[prediction]),
        false
      );
      rows += calculationRow(
        "Logit for " + VOCABULARY[prediction],
        formatPrecise(outputDot) +
          " + " +
          formatPrecise(MODEL.outputBias[prediction]) +
          " = " +
          formatPrecise(predictionLogit),
        true
      );
      rows += calculationRow(
        "Softmax probability",
        (probability * 100).toFixed(4) + "%",
        true
      );
    }

    return (
      '<div class="nextword-lstm__inspector-title">Exact scalar calculation · ' +
      sequence.name +
      " · t" +
      info.timeStep +
      " · " +
      traceSubject +
      "</div>" +
      '<dl class="nextword-lstm__calculation">' +
      rows +
      "</dl>"
    );
  }

  function renderExplanation(info, selectedSentence, selectedFeature, selectedGate) {
    let plain = "";
    if (info.key === "text") {
      plain =
        "<p>Each line is a complete training sequence. The task is to predict the word that follows every current word.</p>";
    } else if (info.key === "tokens") {
      plain =
        "<p>Tokenization turns each sentence into an ordered list. Position matters because every token becomes one training input.</p>";
    } else if (info.key === "ids") {
      plain =
        "<p>A fixed vocabulary gives every token an integer address. The network receives these IDs before the embedding lookup.</p>";
    } else if (info.key === "shift") {
      plain =
        "<p>The input and target are the same sequence shifted by one position. After <code>cats</code>, the correct target is <code>chase</code>. The final real word targets <code>&lt;EOS&gt;</code>.</p>";
    } else if (info.key === "padding") {
      plain =
        "<p>Batch rows need equal length. S1 therefore receives a PAD input and PAD target at t3. The LSTM still calculates that input timestep, but the mask removes its target from loss and accuracy.</p>";
    } else if (info.key === "embedding") {
      plain =
        "<p>Every token ID selects a learned vector with three values. PAD selects the fixed zero vector.</p>";
    } else if (info.key === "metrics") {
      plain =
        "<p>Cross-entropy averages only the eleven positions whose targets are not PAD. The padded S1 target contributes neither loss nor correctness.</p>";
    } else if (info.operation === 0) {
      plain =
        "<p>At t" +
        info.timeStep +
        ", one embedding from each sequence forms <code>x_t</code>. Both <code>h_(t−1)</code> and <code>c_(t−1)</code> also arrive from the previous cell.</p>";
    } else if (info.operation === 1) {
      plain =
        "<p>The current embedding is projected once into sixteen values. Those values are arranged as four consecutive blocks: input, forget, candidate, and output.</p>";
    } else if (info.operation === 2) {
      plain =
        "<p>The previous hidden state is projected into the same sixteen slots. This path lets prior context influence every gate.</p>";
    } else if (info.operation === 3) {
      plain =
        "<p>Input contribution, hidden contribution, and two PyTorch bias vectors are added. The result is split in exact <code>i, f, g, o</code> order. These are still unbounded scores, not fractions or memory updates.</p>" +
        "<p><strong>Four jobs:</strong> <code>i</code> prepares write permission, <code>f</code> prepares old-memory retention, <code>g</code> proposes signed content, and <code>o</code> prepares hidden-state visibility.</p>" +
        "<p><strong>Selected " +
        escapeHTML(selectedGate) +
        " · " +
        escapeHTML(GATE_ROLES[selectedGate]) +
        ":</strong> " +
        escapeHTML(GATE_NOTES[selectedGate].raw) +
        "</p>";
    } else if (info.operation === 4) {
      const activatedValue =
        WALKTHROUGH.records[info.timeStep].gates[selectedGate][
          selectedSentence
        ][selectedFeature];
      plain =
        "<p><strong>Input gate <code>i</code> — write permission:</strong> sigmoid gives 0 to 1. Zero blocks the candidate; one accepts it fully.</p>" +
        "<p><strong>Forget gate <code>f</code> — retention:</strong> sigmoid gives 0 to 1. Zero removes old memory; one keeps it.</p>" +
        "<p><strong>Candidate <code>g</code> — proposed content:</strong> tanh gives −1 to 1. It is a signed proposal rather than a sigmoid control gate.</p>" +
        "<p><strong>Output gate <code>o</code> — visibility:</strong> sigmoid gives 0 to 1. It controls how much of <code>tanh(c_t)</code> becomes <code>h_t</code>, without erasing <code>c_t</code>.</p>" +
        "<p><strong>Selected " +
        escapeHTML(selectedGate) +
        " · " +
        escapeHTML(GATE_ROLES[selectedGate]) +
        ":</strong> " +
        activatedGateEffect(selectedGate, activatedValue) +
        "</p>";
    } else if (info.operation === 5) {
      plain =
        "<p>The forget gate answers “how much old memory remains?” through <code>f_t ⊙ c_(t−1)</code>. The candidate <code>g_t</code> proposes signed new content, and the input gate answers “how much gets written?” through <code>i_t ⊙ g_t</code>.</p>" +
        "<p>The two paths are added: <code>c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t</code>. Retention and writing are separate decisions for every memory feature.</p>";
    } else if (info.operation === 6) {
      plain =
        "<p>The output gate answers “how much updated memory becomes visible now?” through <code>h_t = o_t ⊙ tanh(c_t)</code>. A value near 0 hides that feature from <code>h_t</code>; a value near 1 exposes most of its bounded memory value.</p>" +
        "<p>This gate does not delete <code>c_t</code>. Cell memory continues internally, while hidden state is sent to the vocabulary head and the next timestep.</p>";
    } else {
      const target =
        DATASET[selectedSentence].targetIDs[info.timeStep];
      plain =
        "<p>The hidden state is projected to fourteen vocabulary logits. Stable softmax converts them to probabilities. The selected target is <code>" +
        escapeHTML(VOCABULARY[target]) +
        "</code>" +
        (target === 0
          ? ", so this position is calculated but ignored by the loss."
          : ".") +
        "</p>";
    }
    return {
      plain: plain,
      inspector: scalarInspector(
        info,
        selectedSentence,
        selectedFeature,
        selectedGate
      ),
    };
  }

  function renderMetrics() {
    const predictionRows = [];
    DATASET.forEach(function (item, sequenceIndex) {
      for (let timeStep = 0; timeStep < T; timeStep += 1) {
        const target = item.targetIDs[timeStep];
        const prediction =
          WALKTHROUGH.predictions[sequenceIndex][timeStep];
        predictionRows.push({
          label: item.name + " t" + timeStep,
          value:
            VOCABULARY[prediction] +
            "  | target " +
            VOCABULARY[target] +
            (target === 0 ? "  | MASKED OUT" : ""),
        });
      }
    });
    return (
      '<div class="nextword-lstm__metric-grid">' +
      "<div><strong>Valid targets</strong><code>" +
      WALKTHROUGH.validCount +
      " of " +
      B * T +
      " positions</code></div>" +
      "<div><strong>Correct valid predictions</strong><code>" +
      WALKTHROUGH.correct +
      "/" +
      WALKTHROUGH.validCount +
      " = " +
      (100 * WALKTHROUGH.accuracy).toFixed(1) +
      "%</code></div>" +
      "<div><strong>Masked mean cross-entropy</strong><code>" +
      formatPrecise(WALKTHROUGH.maskedLoss) +
      "</code></div>" +
      "<div><strong>Ignored position</strong><code>S1 · t3 · PAD target</code></div>" +
      "</div>" +
      '<div class="nextword-lstm__metric-predictions">' +
      dataList(predictionRows) +
      "</div>"
    );
  }

  function stageStatus(info, position) {
    const prefix =
      "Operation " + (position + 1) + " of " + (FINAL_POSITION + 1) + " · ";
    if (info.kind === "lstm") {
      return (
        prefix +
        "LSTM · t=" +
        info.timeStep +
        " · " +
        CELL_OPERATIONS[info.operation]
      );
    }
    return prefix + info.title;
  }

  function initialize(container, instanceIndex) {
    if (container.dataset.nextwordLstmReady === "true") return;
    container.dataset.nextwordLstmReady = "true";

    const shell = container.closest("[data-nextword-lstm-shell]");
    const openButton = container.querySelector(
      "[data-nextword-lstm-fullscreen-open]"
    );
    const closeButton = container.querySelector(
      "[data-nextword-lstm-fullscreen-close]"
    );
    const titleElement = container.querySelector(".nextword-lstm__title");
    const statusElement = container.querySelector(
      "[data-nextword-lstm-status]"
    );
    const progressElement = container.querySelector(
      "[data-nextword-lstm-progress]"
    );
    const stageNavElement = container.querySelector(
      "[data-nextword-lstm-stage-nav]"
    );
    const timestepNavElement = container.querySelector(
      "[data-nextword-lstm-timestep-nav]"
    );
    const operationNavElement = container.querySelector(
      "[data-nextword-lstm-operation-nav]"
    );
    const sentencePickerElement = container.querySelector(
      "[data-nextword-lstm-sentence-picker]"
    );
    const featurePickerElement = container.querySelector(
      "[data-nextword-lstm-feature-picker]"
    );
    const gatePickerElement = container.querySelector(
      "[data-nextword-lstm-gate-picker]"
    );
    const batchElement = container.querySelector("[data-nextword-lstm-batch]");
    const cellMapElement = container.querySelector(
      "[data-nextword-lstm-cell-map]"
    );
    const matricesElement = container.querySelector(
      "[data-nextword-lstm-matrices]"
    );
    const metricsElement = container.querySelector(
      "[data-nextword-lstm-metrics]"
    );
    const explanationElement = container.querySelector(
      "[data-nextword-lstm-explanation]"
    );
    const inspectorElement = container.querySelector(
      "[data-nextword-lstm-inspector]"
    );
    const scrubber = container.querySelector(
      "[data-nextword-lstm-scrubber]"
    );
    const playButton = container.querySelector(
      '[data-nextword-lstm-action="play"]'
    );
    const previousButton = container.querySelector(
      '[data-nextword-lstm-action="previous"]'
    );
    const nextButton = container.querySelector(
      '[data-nextword-lstm-action="next"]'
    );
    const resetButton = container.querySelector(
      '[data-nextword-lstm-action="reset"]'
    );
    const speedControl = container.querySelector(
      "[data-nextword-lstm-speed]"
    );
    const tabs = Array.from(
      container.querySelectorAll("[data-nextword-lstm-tab]")
    );
    const panels = Array.from(
      container.querySelectorAll("[data-nextword-lstm-panel]")
    );

    const titleID = "nextword-lstm-title-" + instanceIndex;
    titleElement.id = titleID;
    tabs.forEach(function (tab) {
      const name = tab.dataset.nextwordLstmTab;
      const panel = container.querySelector(
        '[data-nextword-lstm-panel="' + name + '"]'
      );
      const tabID = "nextword-lstm-tab-" + instanceIndex + "-" + name;
      const panelID = "nextword-lstm-panel-" + instanceIndex + "-" + name;
      tab.id = tabID;
      tab.setAttribute("aria-controls", panelID);
      panel.id = panelID;
      panel.setAttribute("aria-labelledby", tabID);
    });

    let position = 0;
    let selectedSentence = 0;
    let selectedFeature = 0;
    let selectedGate = "i";
    let activeTab = "journey";
    let timer = null;
    let isFullscreen = false;
    let restoreFocusElement = null;
    const fullscreenController = { close: closeFullscreen };

    function focusWithoutScroll(element) {
      if (!element) return;
      try {
        element.focus({ preventScroll: true });
      } catch (_error) {
        element.focus();
      }
    }

    function visibleFocusableElements() {
      return Array.from(
        container.querySelectorAll(
          'a[href], button:not([disabled]), select:not([disabled]), ' +
            'input:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
      ).filter(function (element) {
        return (
          !element.hidden &&
          element.getAttribute("aria-hidden") !== "true" &&
          element.getClientRects().length > 0
        );
      });
    }

    function handleFullscreenKeydown(event) {
      if (!isFullscreen) return;
      if (event.key === "Escape") {
        event.preventDefault();
        closeFullscreen();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = visibleFocusableElements();
      if (focusable.length === 0) {
        event.preventDefault();
        focusWithoutScroll(container);
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (!container.contains(active)) {
        event.preventDefault();
        focusWithoutScroll(first);
      } else if (event.shiftKey && active === first) {
        event.preventDefault();
        focusWithoutScroll(last);
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        focusWithoutScroll(first);
      }
    }

    function openFullscreen() {
      if (isFullscreen) return;
      if (
        activeFullscreenController &&
        activeFullscreenController !== fullscreenController
      ) {
        activeFullscreenController.close({ restoreFocus: false });
      }
      restoreFocusElement = document.activeElement;
      isFullscreen = true;
      activeFullscreenController = fullscreenController;
      shell.classList.add("is-fullscreen");
      container.classList.add("is-fullscreen");
      container.setAttribute("role", "dialog");
      container.setAttribute("aria-modal", "true");
      container.setAttribute("aria-labelledby", titleID);
      openButton.setAttribute("aria-expanded", "true");
      closeButton.hidden = false;
      document.documentElement.classList.add("nextword-lstm-modal-open");
      document.body.classList.add("nextword-lstm-modal-open");
      document.addEventListener("keydown", handleFullscreenKeydown);
      container.scrollTop = 0;
      window.requestAnimationFrame(function () {
        focusWithoutScroll(closeButton);
      });
    }

    function closeFullscreen(options) {
      if (!isFullscreen) return;
      const shouldRestore = !options || options.restoreFocus !== false;
      isFullscreen = false;
      stop();
      shell.classList.remove("is-fullscreen");
      container.classList.remove("is-fullscreen");
      container.removeAttribute("role");
      container.removeAttribute("aria-modal");
      container.removeAttribute("aria-labelledby");
      openButton.setAttribute("aria-expanded", "false");
      closeButton.hidden = true;
      document.removeEventListener("keydown", handleFullscreenKeydown);
      if (activeFullscreenController === fullscreenController) {
        activeFullscreenController = null;
        document.documentElement.classList.remove(
          "nextword-lstm-modal-open"
        );
        document.body.classList.remove("nextword-lstm-modal-open");
      }
      if (
        shouldRestore &&
        restoreFocusElement &&
        document.contains(restoreFocusElement)
      ) {
        focusWithoutScroll(restoreFocusElement);
      }
      restoreFocusElement = null;
    }

    function activateTab(name, focus) {
      activeTab = name;
      tabs.forEach(function (tab) {
        const active = tab.dataset.nextwordLstmTab === activeTab;
        tab.classList.toggle("is-current", active);
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
        if (active && focus) tab.focus();
      });
      panels.forEach(function (panel) {
        panel.hidden = panel.dataset.nextwordLstmPanel !== activeTab;
      });
    }

    function stop() {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
      playButton.textContent = "Play";
      playButton.setAttribute("aria-pressed", "false");
      statusElement.setAttribute("aria-live", "polite");
    }

    function render() {
      const info = positionInfo(position);
      const explanation = renderExplanation(
        info,
        selectedSentence,
        selectedFeature,
        selectedGate
      );
      if (position === FINAL_POSITION) {
        statusElement.setAttribute("aria-live", "polite");
      }
      statusElement.textContent = stageStatus(info, position);
      progressElement.style.width =
        (((position + 1) / (FINAL_POSITION + 1)) * 100).toFixed(2) + "%";
      previousButton.disabled = position === 0;
      nextButton.disabled = position === FINAL_POSITION;
      scrubber.value = String(position);
      scrubber.setAttribute("aria-valuetext", stageStatus(info, position));

      stageNavElement.innerHTML = renderStageNavigation(position);
      const currentStage = stageNavElement.querySelector(
        '[aria-current="step"]'
      );
      if (currentStage) {
        stageNavElement.scrollLeft = Math.max(
          0,
          currentStage.offsetLeft -
            (stageNavElement.clientWidth - currentStage.offsetWidth) / 2
        );
      }
      timestepNavElement.innerHTML = renderTimestepNavigation(info);
      operationNavElement.innerHTML = renderOperationNavigation(info);
      sentencePickerElement.innerHTML =
        renderSentencePicker(selectedSentence);
      featurePickerElement.innerHTML =
        renderFeaturePicker(selectedFeature);
      gatePickerElement.innerHTML = renderGatePicker(selectedGate);
      batchElement.innerHTML = renderBatch(info, selectedSentence);
      cellMapElement.innerHTML = renderCellMap(info, selectedGate);

      if (info.kind === "pre") {
        matricesElement.innerHTML = renderPreMatrices(
          info,
          selectedSentence
        );
      } else if (info.kind === "lstm") {
        matricesElement.innerHTML = renderLSTMMatrices(
          info,
          selectedSentence,
          selectedFeature,
          selectedGate
        );
      } else {
        matricesElement.innerHTML = "";
      }

      explanationElement.innerHTML = explanation.plain;
      inspectorElement.innerHTML = explanation.inspector;
      metricsElement.hidden = info.kind !== "metrics";
      metricsElement.innerHTML =
        info.kind === "metrics" ? renderMetrics() : "";
      if (position === FINAL_POSITION) stop();
    }

    function jumpTo(nextPosition) {
      stop();
      position = Math.max(0, Math.min(FINAL_POSITION, nextPosition));
      render();
    }

    function play() {
      if (timer !== null) {
        stop();
        return;
      }
      if (position >= FINAL_POSITION) {
        position = 0;
        render();
      }
      playButton.textContent = "Pause";
      playButton.setAttribute("aria-pressed", "true");
      statusElement.setAttribute("aria-live", "off");
      timer = window.setInterval(function () {
        if (position < FINAL_POSITION) {
          position += 1;
          render();
        } else {
          stop();
        }
      }, Number(speedControl.value));
    }

    resetButton.addEventListener("click", function () { jumpTo(0); });
    previousButton.addEventListener("click", function () {
      jumpTo(position - 1);
    });
    nextButton.addEventListener("click", function () {
      jumpTo(position + 1);
    });
    playButton.addEventListener("click", play);
    speedControl.addEventListener("change", function () {
      if (timer !== null) {
        stop();
        play();
      }
    });
    scrubber.addEventListener("input", function () {
      jumpTo(Number(scrubber.value));
    });
    openButton.addEventListener("click", openFullscreen);
    closeButton.addEventListener("click", function () {
      closeFullscreen();
    });
    shell.addEventListener("click", function (event) {
      if (isFullscreen && event.target === shell) closeFullscreen();
    });

    container.addEventListener("click", function (event) {
      const tab = event.target.closest("[data-nextword-lstm-tab]");
      if (tab && container.contains(tab)) {
        activateTab(tab.dataset.nextwordLstmTab, false);
        return;
      }
      const jump = event.target.closest("[data-nextword-lstm-jump]");
      if (jump && container.contains(jump)) {
        const jumpValue = jump.dataset.nextwordLstmJump;
        const navigationHost = jump.closest(
          "[data-nextword-lstm-stage-nav], " +
            "[data-nextword-lstm-timestep-nav], " +
            "[data-nextword-lstm-operation-nav]"
        );
        let replacementHost = stageNavElement;
        if (
          navigationHost &&
          navigationHost.hasAttribute("data-nextword-lstm-timestep-nav")
        ) {
          replacementHost = timestepNavElement;
        } else if (
          navigationHost &&
          navigationHost.hasAttribute("data-nextword-lstm-operation-nav")
        ) {
          replacementHost = operationNavElement;
        }
        jumpTo(Number(jumpValue));
        focusWithoutScroll(
          replacementHost.querySelector(
            '[data-nextword-lstm-jump="' + jumpValue + '"]'
          )
        );
        return;
      }
      const sentence = event.target.closest(
        "[data-nextword-lstm-select-sentence]"
      );
      if (sentence && container.contains(sentence)) {
        const sentenceValue =
          sentence.dataset.nextwordLstmSelectSentence;
        const cameFromPicker = Boolean(
          sentence.closest("[data-nextword-lstm-sentence-picker]")
        );
        selectedSentence = Number(sentenceValue);
        stop();
        render();
        focusWithoutScroll(
          (cameFromPicker ? sentencePickerElement : batchElement).querySelector(
            '[data-nextword-lstm-select-sentence="' +
              sentenceValue +
              '"]'
          )
        );
        return;
      }
      const feature = event.target.closest(
        "[data-nextword-lstm-select-feature]"
      );
      if (feature && container.contains(feature)) {
        const featureValue =
          feature.dataset.nextwordLstmSelectFeature;
        selectedFeature = Number(featureValue);
        stop();
        render();
        focusWithoutScroll(
          featurePickerElement.querySelector(
            '[data-nextword-lstm-select-feature="' +
              featureValue +
              '"]'
          )
        );
        return;
      }
      const gate = event.target.closest(
        "[data-nextword-lstm-select-gate]"
      );
      if (gate && container.contains(gate)) {
        const gateValue = gate.dataset.nextwordLstmSelectGate;
        selectedGate = gateValue;
        stop();
        render();
        focusWithoutScroll(
          gatePickerElement.querySelector(
            '[data-nextword-lstm-select-gate="' + gateValue + '"]'
          )
        );
      }
    });

    container.addEventListener("keydown", function (event) {
      const tab = event.target.closest("[data-nextword-lstm-tab]");
      if (
        tab &&
        ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
      ) {
        event.preventDefault();
        const currentIndex = tabs.indexOf(tab);
        let nextIndex = currentIndex;
        if (event.key === "ArrowLeft") {
          nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
        } else if (event.key === "ArrowRight") {
          nextIndex = (currentIndex + 1) % tabs.length;
        } else if (event.key === "Home") {
          nextIndex = 0;
        } else if (event.key === "End") {
          nextIndex = tabs.length - 1;
        }
        activateTab(tabs[nextIndex].dataset.nextwordLstmTab, true);
        return;
      }
      if (event.target.matches("button, input, select, a")) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        jumpTo(position - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        jumpTo(position + 1);
      } else if (event.key === "Home") {
        event.preventDefault();
        jumpTo(0);
      } else if (event.key === "End") {
        event.preventDefault();
        jumpTo(FINAL_POSITION);
      } else if (event.key === " ") {
        event.preventDefault();
        play();
      }
    });

    activateTab(activeTab, false);
    render();
    container.classList.add("is-ready");
  }

  function deepFreeze(value) {
    if (!value || typeof value !== "object" || Object.isFrozen(value)) {
      return value;
    }
    Object.keys(value).forEach(function (key) {
      deepFreeze(value[key]);
    });
    return Object.freeze(value);
  }

  window.__lstmNextWordTest = deepFreeze({
    dimensions: { B: B, T: T, E: E, H: H, V: V },
    gateOrder: GATE_ORDER.slice(),
    vocabulary: VOCABULARY.slice(),
    dataset: DATASET.map(function (item) {
      return {
        name: item.name,
        text: item.text,
        tokens: item.tokens.slice(),
        inputIDs: item.inputIDs.slice(),
        targetIDs: item.targetIDs.slice(),
        length: item.length,
      };
    }),
    data: {
      inputs: WALKTHROUGH.paddedIDs.map(function (row) { return row.slice(); }),
      targets: WALKTHROUGH.targetIDs.map(function (row) { return row.slice(); }),
      mask: WALKTHROUGH.mask.map(function (row) { return row.slice(); }),
      lengths: WALKTHROUGH.lengths.slice(),
    },
    weights: JSON.parse(JSON.stringify(MODEL)),
    paddedIDs: WALKTHROUGH.paddedIDs.map(function (row) { return row.slice(); }),
    targets: WALKTHROUGH.targetIDs.map(function (row) { return row.slice(); }),
    mask: WALKTHROUGH.mask.map(function (row) { return row.slice(); }),
    embedded: JSON.parse(JSON.stringify(WALKTHROUGH.embedded)),
    records: JSON.parse(JSON.stringify(WALKTHROUGH.records)),
    logits: JSON.parse(JSON.stringify(WALKTHROUGH.logits)),
    probabilities: JSON.parse(JSON.stringify(WALKTHROUGH.probabilities)),
    predictions: WALKTHROUGH.predictions.map(function (row) {
      return row.slice();
    }),
    maskedLoss: WALKTHROUGH.maskedLoss,
    accuracy: WALKTHROUGH.accuracy,
    validCount: WALKTHROUGH.validCount,
    correct: WALKTHROUGH.correct,
    positions: {
      recurrenceStart: RECURRENCE_START,
      metrics: METRICS_POSITION,
      final: FINAL_POSITION,
      count: FINAL_POSITION + 1,
    },
    macroStages: MACRO_STAGES.map(function (stage) {
      return {
        key: stage.key,
        label: stage.label,
        position: stage.position,
      };
    }),
    cellOperations: CELL_OPERATIONS.slice(),
  });

  function boot() {
    document
      .querySelectorAll("[data-nextword-lstm]")
      .forEach(function (container, index) {
        initialize(container, index);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
