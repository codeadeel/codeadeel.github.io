(function () {
  "use strict";

  if (window.__sentenceRNNAnimationLoaded) return;
  window.__sentenceRNNAnimationLoaded = true;

  let activeFullscreenController = null;

  const B = 3;
  const T = 5;
  const E = 3;
  const H = 4;
  const C = 2;

  const PRE_STAGE_COUNT = 5;
  const CELL_OPERATION_COUNT = 5;
  const RECURRENCE_START = PRE_STAGE_COUNT;
  const RECURRENCE_END = RECURRENCE_START + T * CELL_OPERATION_COUNT;
  const LAST_HIDDEN_POSITION = RECURRENCE_END;
  const LOGITS_POSITION = LAST_HIDDEN_POSITION + 1;
  const SOFTMAX_POSITION = LOGITS_POSITION + 1;
  const METRICS_POSITION = SOFTMAX_POSITION + 1;
  const FINAL_POSITION = METRICS_POSITION;

  const CLASS_NAMES = ["negative", "positive"];
  const HIDDEN_NAMES = ["h0", "h1", "h2", "h3"];
  const EMBEDDING_NAMES = ["e0", "e1", "e2"];

  const VOCABULARY = [
    "<PAD>",
    "<UNK>",
    "movie",
    "was",
    "good",
    "acting",
    "very",
    "bad",
    "the",
    "story",
    "is",
    "quite",
    "wonderful",
  ];

  const DATASET = [
    {
      name: "S0",
      text: "movie was good",
      tokens: ["movie", "was", "good"],
      ids: [2, 3, 4],
      label: 1,
    },
    {
      name: "S1",
      text: "acting was very bad",
      tokens: ["acting", "was", "very", "bad"],
      ids: [5, 3, 6, 7],
      label: 0,
    },
    {
      name: "S2",
      text: "the story is quite wonderful",
      tokens: ["the", "story", "is", "quite", "wonderful"],
      ids: [8, 9, 10, 11, 12],
      label: 1,
    },
  ];

  // The browser uses row-batch equations:
  // x_t @ weightIHUsed and h_prev @ weightHHUsed.
  // PyTorch stores weight_ih_l0 and weight_hh_l0 transposed relative to these.
  const MODEL = {
    embedding: [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.1, 0.4, 0.1],
      [0.0, 0.2, 0.1],
      [1.4, 0.2, 0.1],
      [0.05, 0.5, -0.1],
      [0.0, 0.1, 0.0],
      [-1.4, 0.2, -0.1],
      [0.0, 0.05, 0.0],
      [0.1, 0.6, 0.2],
      [0.0, 0.1, 0.0],
      [0.0, 0.1, 0.05],
      [1.6, 0.2, 0.2],
    ], // (V=13,E=3)
    weightIHUsed: [
      [1.0, -0.8, 0.4, 0.2],
      [0.15, 0.25, 0.5, -0.3],
      [-0.2, 0.3, 0.2, 0.6],
    ], // (E=3,H=4), equal to PyTorch weight_ih_l0.T
    weightHHUsed: [
      [0.55, 0.08, -0.04, 0.02],
      [-0.05, 0.45, 0.06, -0.03],
      [0.04, -0.08, 0.5, 0.07],
      [0.02, 0.04, -0.05, 0.4],
    ], // (H=4,H=4), equal to PyTorch weight_hh_l0.T
    biasIH: [0.02, -0.01, 0.0, 0.01],
    biasHH: [-0.01, -0.01, 0.0, 0.0],
    classifierWeightUsed: [
      [-1.3, 1.3],
      [0.2, -0.2],
      [-0.1, 0.1],
      [0.05, -0.05],
    ], // (H=4,C=2), equal to nn.Linear.weight.T
    classifierBias: [0.0, 0.0],
  };

  const MACRO_STAGES = [
    { key: "text", label: "Text", position: 0 },
    { key: "tokens", label: "Tokens", position: 1 },
    { key: "ids", label: "IDs", position: 2 },
    { key: "padding", label: "Padding", position: 3 },
    { key: "embedding", label: "Embedding", position: 4 },
    { key: "recurrence", label: "RNN recurrence", position: RECURRENCE_START },
    {
      key: "lastHidden",
      label: "Last-real hidden",
      position: LAST_HIDDEN_POSITION,
    },
    { key: "logits", label: "Logits", position: LOGITS_POSITION },
    { key: "softmax", label: "Softmax", position: SOFTMAX_POSITION },
    { key: "metrics", label: "Metrics", position: METRICS_POSITION },
  ];

  const CELL_OPERATIONS = [
    "Select x_t",
    "x_t @ W_ihᵀ",
    "h_prev @ W_hhᵀ",
    "Add biases",
    "tanh → h_t",
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

  function addRNNParts(input, memory, biasIH, biasHH) {
    return input.map(function (row, rowIndex) {
      return row.map(function (value, column) {
        return (
          value +
          memory[rowIndex][column] +
          biasIH[column] +
          biasHH[column]
        );
      });
    });
  }

  function addBias(matrix, bias) {
    return matrix.map(function (row) {
      return row.map(function (value, column) {
        return value + bias[column];
      });
    });
  }

  function tanhMatrix(matrix) {
    return matrix.map(function (row) {
      return row.map(Math.tanh);
    });
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
    return row.reduce(function (bestIndex, value, index, values) {
      return value > values[bestIndex] ? index : bestIndex;
    }, 0);
  }

  function computeWalkthrough() {
    const lengths = DATASET.map(function (item) {
      return item.ids.length;
    });
    const labels = DATASET.map(function (item) {
      return item.label;
    });
    const paddedIDs = DATASET.map(function (item) {
      return item.ids.concat(Array(T - item.ids.length).fill(0));
    });
    const embedded = paddedIDs.map(function (row) {
      return row.map(function (id) {
        return MODEL.embedding[id].slice();
      });
    });

    let hidden = zeros(B, H);
    const records = [];
    const hiddenHistory = Array.from({ length: B }, function () {
      return [];
    });

    for (let timeStep = 0; timeStep < T; timeStep += 1) {
      const xT = embedded.map(function (sentence) {
        return sentence[timeStep].slice();
      });
      const hiddenPrevious = cloneMatrix(hidden);
      const inputProduct = matMul(xT, MODEL.weightIHUsed);
      const memoryProduct = matMul(hiddenPrevious, MODEL.weightHHUsed);
      const preActivation = addRNNParts(
        inputProduct,
        memoryProduct,
        MODEL.biasIH,
        MODEL.biasHH
      );
      const newHidden = tanhMatrix(preActivation);

      records.push({
        timeStep: timeStep,
        xT: xT,
        hiddenPrevious: hiddenPrevious,
        inputProduct: inputProduct,
        memoryProduct: memoryProduct,
        preActivation: preActivation,
        newHidden: newHidden,
      });

      newHidden.forEach(function (row, sentenceIndex) {
        hiddenHistory[sentenceIndex].push(row.slice());
      });
      hidden = newHidden;
    }

    const lastIndices = lengths.map(function (length) {
      return length - 1;
    });
    const lastRealHidden = hiddenHistory.map(function (history, row) {
      return history[lastIndices[row]].slice();
    });
    const logits = addBias(
      matMul(lastRealHidden, MODEL.classifierWeightUsed),
      MODEL.classifierBias
    );
    const probabilities = softmaxRows(logits);
    const predictions = probabilities.map(argmax);
    const rowLosses = labels.map(function (label, row) {
      return -Math.log(Math.max(probabilities[row][label], 1e-12));
    });
    const loss =
      rowLosses.reduce(function (sum, value) {
        return sum + value;
      }, 0) / B;
    const correct = predictions.reduce(function (sum, prediction, row) {
      return sum + Number(prediction === labels[row]);
    }, 0);

    return {
      lengths: lengths,
      labels: labels,
      paddedIDs: paddedIDs,
      embedded: embedded,
      records: records,
      hiddenHistory: hiddenHistory,
      paddedFinalHidden: hidden,
      lastIndices: lastIndices,
      lastRealHidden: lastRealHidden,
      logits: logits,
      probabilities: probabilities,
      predictions: predictions,
      rowLosses: rowLosses,
      loss: loss,
      accuracy: correct / B,
      correct: correct,
    };
  }

  const WALKTHROUGH = computeWalkthrough();

  function formatNumber(value) {
    if (Object.is(value, -0) || Math.abs(value) < 0.005) return "0.00";
    return value.toFixed(2);
  }

  function formatPrecise(value) {
    if (Object.is(value, -0) || Math.abs(value) < 0.00005) return "0.0000";
    return value.toFixed(4);
  }

  function escapeHTML(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function cellColor(value, matrix) {
    let maximum = 0;
    matrix.forEach(function (row) {
      row.forEach(function (candidate) {
        maximum = Math.max(maximum, Math.abs(candidate));
      });
    });
    const strength = maximum === 0 ? 0 : Math.min(Math.abs(value) / maximum, 1);
    const alpha = 0.07 + 0.25 * strength;
    return value >= 0
      ? "rgba(37, 99, 235, " + alpha.toFixed(3) + ")"
      : "rgba(214, 69, 69, " + alpha.toFixed(3) + ")";
  }

  function matrixTable(matrix, options) {
    const settings = options || {};
    const rowLabels = settings.rowLabels || [];
    const columnLabels = settings.columnLabels || [];
    const selectedRow = settings.selectedRow;
    const selectedColumn = settings.selectedColumn;
    const tableLabel = settings.label
      ? ' aria-label="' + escapeHTML(settings.label) + '"'
      : "";

    const header =
      columnLabels.length === 0
        ? ""
        : "<thead><tr>" +
          (rowLabels.length > 0 ? '<th scope="col"></th>' : "") +
          columnLabels
            .map(function (label, column) {
              return (
                '<th scope="col"' +
                (column === selectedColumn
                  ? ' class="is-selected-column"'
                  : "") +
                ">" +
                escapeHTML(label) +
                "</th>"
              );
            })
            .join("") +
          "</tr></thead>";

    const rows = matrix
      .map(function (row, rowIndex) {
        const cells = row
          .map(function (value, column) {
            return (
              "<td" +
              (column === selectedColumn
                ? ' class="is-selected-column"'
                : "") +
              ' title="' +
              Number(value).toFixed(6) +
              '" style="background:' +
              cellColor(Number(value), matrix) +
              '">' +
              formatNumber(Number(value)) +
              "</td>"
            );
          })
          .join("");
        return (
          "<tr" +
          (rowIndex === selectedRow ? ' class="is-selected"' : "") +
          ">" +
          (rowLabels.length > 0
            ? '<th scope="row">' + escapeHTML(rowLabels[rowIndex]) + "</th>"
            : "") +
          cells +
          "</tr>"
        );
      })
      .join("");

    return (
      '<div class="sentence-rnn__matrix-scroll"><table class="sentence-rnn__matrix"' +
      tableLabel +
      ">" +
      header +
      "<tbody>" +
      rows +
      "</tbody></table></div>"
    );
  }

  function matrixCard(title, shape, matrix, note, active, options) {
    const tableOptions = Object.assign({}, options || {}, {
      label: title + " " + shape,
    });
    return (
      '<section class="sentence-rnn__matrix-card' +
      (active ? " sentence-rnn__matrix-card--active" : "") +
      '">' +
      '<div class="sentence-rnn__matrix-title">' +
      escapeHTML(title) +
      " · " +
      escapeHTML(shape) +
      "</div>" +
      matrixTable(matrix, tableOptions) +
      (note
        ? '<div class="sentence-rnn__matrix-note">' + note + "</div>"
        : "") +
      "</section>"
    );
  }

  function informationCard(title, shape, body, active) {
    return (
      '<section class="sentence-rnn__matrix-card sentence-rnn__information-card' +
      (active ? " sentence-rnn__matrix-card--active" : "") +
      '">' +
      '<div class="sentence-rnn__matrix-title">' +
      escapeHTML(title) +
      " · " +
      escapeHTML(shape) +
      "</div>" +
      body +
      "</section>"
    );
  }

  function listRows(items) {
    return (
      '<div class="sentence-rnn__data-list">' +
      items
        .map(function (item) {
          return (
            '<div><strong>' +
            escapeHTML(item.name) +
            "</strong><code>" +
            escapeHTML(item.value) +
            "</code></div>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function positionInfo(position) {
    if (position < PRE_STAGE_COUNT) {
      return {
        kind: "pre",
        key: MACRO_STAGES[position].key,
        macroIndex: position,
        title: MACRO_STAGES[position].label,
      };
    }
    if (position < RECURRENCE_END) {
      const relative = position - RECURRENCE_START;
      return {
        kind: "rnn",
        key: "recurrence",
        macroIndex: 5,
        timeStep: Math.floor(relative / CELL_OPERATION_COUNT),
        operation: relative % CELL_OPERATION_COUNT,
        title: CELL_OPERATIONS[relative % CELL_OPERATION_COUNT],
      };
    }
    const macroIndex = 6 + (position - LAST_HIDDEN_POSITION);
    return {
      kind: "post",
      key: MACRO_STAGES[macroIndex].key,
      macroIndex: macroIndex,
      title: MACRO_STAGES[macroIndex].label,
    };
  }

  function renderStageNavigation(position) {
    const currentInfo = positionInfo(position);
    return MACRO_STAGES.map(function (stage, index) {
      const complete = index < currentInfo.macroIndex;
      const current = index === currentInfo.macroIndex;
      const button =
        '<button type="button" class="' +
        (current ? "is-current" : complete ? "is-complete" : "") +
        '" data-sentence-rnn-jump="' +
        stage.position +
        '"' +
        (current ? ' aria-current="step"' : "") +
        ' aria-label="Stage ' +
        (index + 1) +
        " of " +
        MACRO_STAGES.length +
        ": " +
        escapeHTML(stage.label) +
        '">' +
        '<span class="sentence-rnn__stage-number" aria-hidden="true">' +
        (index + 1) +
        "</span>" +
        '<span class="sentence-rnn__stage-label">' +
        escapeHTML(stage.label) +
        "</span>" +
        "</button>";
      const connector =
        index < MACRO_STAGES.length - 1
          ? '<span class="sentence-rnn__stage-connector' +
            (index < currentInfo.macroIndex ? " is-complete" : "") +
            '" aria-hidden="true"><span></span></span>'
          : "";
      return button + connector;
    }).join("");
  }

  function renderTimestepNavigation(info) {
    const currentTime = info.kind === "rnn" ? info.timeStep : -1;
    return Array.from({ length: T }, function (_, timeStep) {
      return (
        '<button type="button" class="' +
        (currentTime === timeStep ? "is-current" : "") +
        '" data-sentence-rnn-jump="' +
        (RECURRENCE_START + timeStep * CELL_OPERATION_COUNT) +
        '"' +
        (currentTime === timeStep ? ' aria-current="step"' : "") +
        ">t" +
        timeStep +
        "</button>"
      );
    }).join("");
  }

  function renderOperationNavigation(info) {
    const currentTime = info.kind === "rnn" ? info.timeStep : 0;
    const currentOperation = info.kind === "rnn" ? info.operation : -1;
    return CELL_OPERATIONS.map(function (operation, index) {
      return (
        '<button type="button" class="' +
        (currentOperation === index ? "is-current" : "") +
        '" data-sentence-rnn-jump="' +
        (RECURRENCE_START +
          currentTime * CELL_OPERATION_COUNT +
          index) +
        '"' +
        (currentOperation === index ? ' aria-current="step"' : "") +
        ">" +
        escapeHTML(operation) +
        "</button>"
      );
    }).join("");
  }

  function renderSentencePicker(selectedSentence) {
    return DATASET.map(function (item, index) {
      return (
        '<button type="button" class="' +
        (index === selectedSentence ? "is-current" : "") +
        '" data-sentence-rnn-select-sentence="' +
        index +
        '" aria-pressed="' +
        String(index === selectedSentence) +
        '">' +
        item.name +
        " · y=" +
        item.label +
        "</button>"
      );
    }).join("");
  }

  function renderFeaturePicker(selectedFeature) {
    return HIDDEN_NAMES.map(function (name, index) {
      return (
        '<button type="button" class="' +
        (index === selectedFeature ? "is-current" : "") +
        '" data-sentence-rnn-select-feature="' +
        index +
        '" aria-pressed="' +
        String(index === selectedFeature) +
        '">' +
        name +
        "</button>"
      );
    }).join("");
  }

  function renderBatch(info, selectedSentence) {
    const showTokens = info.macroIndex >= 1;
    const showIDs = info.macroIndex >= 2;
    const showPadding = info.macroIndex >= 3;
    const currentTime = info.kind === "rnn" ? info.timeStep : -1;
    const recurrenceComplete = info.macroIndex > 5;

    return DATASET.map(function (item, row) {
      const paddedTokens = item.tokens.concat(
        Array(T - item.tokens.length).fill("<PAD>")
      );
      const visibleTokens = showPadding ? paddedTokens : item.tokens;
      const visibleIDs = showPadding ? WALKTHROUGH.paddedIDs[row] : item.ids;
      const completedRealTokens = recurrenceComplete
        ? item.tokens.length
        : info.kind === "rnn"
          ? Math.min(currentTime, item.tokens.length)
          : 0;
      const progressText = info.key === "lastHidden"
        ? "last real word selected"
        : recurrenceComplete
          ? "sentence memory ready"
        : info.kind === "rnn"
          ? completedRealTokens +
            " of " +
            item.tokens.length +
            " words remembered"
          : showTokens
            ? item.tokens.length + " words ready"
            : "raw sentence";
      const progressPercent =
        item.tokens.length === 0
          ? 0
          : (completedRealTokens / item.tokens.length) * 100;

      let content =
        '<div class="sentence-rnn__batch-content">' +
        '<div class="sentence-rnn__batch-text-line">' +
        '<div class="sentence-rnn__batch-text">“' +
        escapeHTML(item.text) +
        "”</div>" +
        '<span class="sentence-rnn__lane-status">' +
        escapeHTML(progressText) +
        "</span>" +
        "</div>";

      if (showTokens) {
        content +=
          '<div class="sentence-rnn__token-strip" role="list" aria-label="' +
          escapeHTML(item.name) +
          ' sequence">' +
          visibleTokens
            .map(function (token, timeStep) {
              const isPadding = timeStep >= item.tokens.length;
              const isEndpoint =
                info.key === "lastHidden" &&
                timeStep === item.tokens.length - 1;
              const isCurrent = currentTime === timeStep || isEndpoint;
              const isRead =
                timeStep < item.tokens.length &&
                (recurrenceComplete ||
                  (info.kind === "rnn" && timeStep < currentTime));
              const tokenState = isCurrent
                ? isPadding
                  ? "current PAD"
                  : "current word"
                : isRead
                  ? "remembered"
                  : isPadding
                    ? "padding"
                    : "waiting";
              return (
                '<span role="listitem" class="sentence-rnn__token' +
                (isPadding ? " is-padding" : "") +
                (isCurrent ? " is-current-token" : "") +
                (isRead ? " is-read" : "") +
                '" aria-label="timestep ' +
                timeStep +
                ", " +
                escapeHTML(token) +
                ", " +
                tokenState +
                '"' +
                (isCurrent ? ' aria-current="step"' : "") +
                ">" +
                '<span class="sentence-rnn__token-position">t' +
                timeStep +
                "</span>" +
                '<span class="sentence-rnn__token-word">' +
                escapeHTML(token) +
                "</span>" +
                (showIDs
                  ? '<span class="sentence-rnn__token-id">id ' +
                    visibleIDs[timeStep] +
                    "</span>"
                  : "") +
                '<span class="sentence-rnn__token-state" aria-hidden="true">' +
                (isCurrent
                  ? isPadding
                    ? "PAD now"
                    : "current"
                  : isRead
                    ? "remembered"
                    : isPadding
                      ? "PAD"
                      : "waiting") +
                "</span>" +
                "</span>"
              );
            })
            .join("") +
          "</div>";
      }
      content +=
        '<span class="sentence-rnn__lane-progress" aria-hidden="true"><span style="width:' +
        progressPercent.toFixed(2) +
        '%"></span></span></div>';

      return (
        '<button type="button" class="sentence-rnn__batch-row' +
        (row === selectedSentence ? " is-selected" : "") +
        '" data-sentence-rnn-select-sentence="' +
        row +
        '" aria-pressed="' +
        String(row === selectedSentence) +
        '">' +
        '<div class="sentence-rnn__batch-meta"><strong>' +
        item.name +
        "</strong><span>length " +
        item.tokens.length +
        '</span><span class="sentence-rnn__label sentence-rnn__label--' +
        item.label +
        '">y=' +
        item.label +
        " · " +
        CLASS_NAMES[item.label] +
        "</span></div>" +
        content +
        "</button>"
      );
    }).join("");
  }

  function renderPreMatrices(info, selectedSentence) {
    if (info.key === "text") {
      return informationCard(
        "Raw input batch",
        "3 Python strings; ragged, not a tensor",
        listRows(
          DATASET.map(function (item) {
            return { name: item.name, value: '"' + item.text + '"' };
          })
        ),
        true
      );
    }

    if (info.key === "tokens") {
      return [
        informationCard(
          "Tokenized sentences",
          "3 variable-length lists; not a tensor",
          listRows(
            DATASET.map(function (item) {
              return {
                name: item.name,
                value: "[" + item.tokens.join(", ") + "]",
              };
            })
          ),
          true
        ),
        matrixCard(
          "Lengths",
          "(B,) = (3,)",
          [WALKTHROUGH.lengths],
          "One real-token count per sentence.",
          false,
          { columnLabels: ["S0", "S1", "S2"] }
        ),
      ].join("");
    }

    if (info.key === "ids") {
      return [
        informationCard(
          "Integer-ID sequences",
          "3 variable-length lists; not a tensor",
          listRows(
            DATASET.map(function (item) {
              return { name: item.name, value: "[" + item.ids.join(", ") + "]" };
            })
          ),
          true
        ),
        informationCard(
          "Vocabulary lookup",
          "V=13 entries",
          '<div class="sentence-rnn__vocabulary">' +
            VOCABULARY.map(function (word, id) {
              return (
                "<code>" + id + " → " + escapeHTML(word) + "</code>"
              );
            }).join("") +
            "</div>",
          false
        ),
      ].join("");
    }

    if (info.key === "padding") {
      return [
        matrixCard(
          "Padded token IDs",
          "(B,T) = (3,5)",
          WALKTHROUGH.paddedIDs,
          "ID 0 is &lt;PAD&gt;. Rows now have equal width.",
          true,
          {
            rowLabels: ["S0", "S1", "S2"],
            columnLabels: ["t0", "t1", "t2", "t3", "t4"],
            selectedRow: selectedSentence,
          }
        ),
        matrixCard(
          "Lengths",
          "(B,) = (3,)",
          [WALKTHROUGH.lengths],
          "Lengths preserve where each real sentence ends.",
          false,
          { columnLabels: ["S0", "S1", "S2"] }
        ),
      ].join("");
    }

    return [
      matrixCard(
        "Embedding table",
        "(V,E) = (13,3)",
        MODEL.embedding,
        "Each ID selects one row. Row 0 is the zero PAD vector.",
        false,
        {
          rowLabels: VOCABULARY.map(function (word, id) {
            return id + ":" + word;
          }),
          columnLabels: EMBEDDING_NAMES,
        }
      ),
      matrixCard(
        "Selected sentence embedding slice",
        "(T,E) = (5,3); full batch is (3,5,3)",
        WALKTHROUGH.embedded[selectedSentence],
        "One 3-number vector per padded position.",
        true,
        {
          rowLabels: ["t0", "t1", "t2", "t3", "t4"],
          columnLabels: EMBEDDING_NAMES,
        }
      ),
    ].join("");
  }

  function batchMatrixOptions(selectedSentence, selectedFeature) {
    return {
      rowLabels: ["S0", "S1", "S2"],
      columnLabels: HIDDEN_NAMES,
      selectedRow: selectedSentence,
      selectedColumn: selectedFeature,
    };
  }

  function renderRNNMatrices(info, selectedSentence, selectedFeature) {
    const record = WALKTHROUGH.records[info.timeStep];
    const options = batchMatrixOptions(selectedSentence, selectedFeature);

    if (info.operation === 0) {
      return [
        matrixCard(
          "Current word vectors x_t",
          "(B,E) = (3,3)",
          record.xT,
          "One embedding vector from each sentence at t=" +
            info.timeStep +
            ".",
          true,
          {
            rowLabels: ["S0", "S1", "S2"],
            columnLabels: EMBEDDING_NAMES,
            selectedRow: selectedSentence,
          }
        ),
        matrixCard(
          "Previous hidden state h_prev",
          "(B,H) = (3,4)",
          record.hiddenPrevious,
          info.timeStep === 0
            ? "Initialized to zeros."
            : "The hidden state produced at t=" + (info.timeStep - 1) + ".",
          false,
          options
        ),
      ].join("");
    }

    if (info.operation === 1) {
      return [
        matrixCard(
          "x_t",
          "(B,E) = (3,3)",
          record.xT,
          "",
          false,
          {
            rowLabels: ["S0", "S1", "S2"],
            columnLabels: EMBEDDING_NAMES,
            selectedRow: selectedSentence,
          }
        ),
        matrixCard(
          "W_ihᵀ used in row-batch math",
          "(E,H) = (3,4)",
          MODEL.weightIHUsed,
          "nn.RNN stores weight_ih_l0 with shape (4,3); this is its transpose.",
          false,
          {
            rowLabels: EMBEDDING_NAMES,
            columnLabels: HIDDEN_NAMES,
            selectedColumn: selectedFeature,
          }
        ),
        matrixCard(
          "Input contribution",
          "(3,3) @ (3,4) = (3,4)",
          record.inputProduct,
          "Each word vector becomes four hidden features.",
          true,
          options
        ),
      ].join("");
    }

    if (info.operation === 2) {
      return [
        matrixCard(
          "h_prev",
          "(B,H) = (3,4)",
          record.hiddenPrevious,
          "",
          false,
          options
        ),
        matrixCard(
          "W_hhᵀ used in row-batch math",
          "(H,H) = (4,4)",
          MODEL.weightHHUsed,
          "nn.RNN stores weight_hh_l0 with shape (4,4); row-batch math uses its transpose.",
          false,
          {
            rowLabels: HIDDEN_NAMES,
            columnLabels: HIDDEN_NAMES,
            selectedColumn: selectedFeature,
          }
        ),
        matrixCard(
          "Memory contribution",
          "(3,4) @ (4,4) = (3,4)",
          record.memoryProduct,
          "The previous memory is remixed into four new features.",
          true,
          options
        ),
      ].join("");
    }

    if (info.operation === 3) {
      return [
        matrixCard(
          "Input contribution",
          "(B,H) = (3,4)",
          record.inputProduct,
          "",
          false,
          options
        ),
        matrixCard(
          "Memory contribution",
          "(B,H) = (3,4)",
          record.memoryProduct,
          "",
          false,
          options
        ),
        matrixCard(
          "Input bias b_ih",
          "(H,) = (4,)",
          [MODEL.biasIH],
          "Broadcast across all three rows.",
          false,
          {
            rowLabels: ["b_ih"],
            columnLabels: HIDDEN_NAMES,
            selectedColumn: selectedFeature,
          }
        ),
        matrixCard(
          "Memory bias b_hh",
          "(H,) = (4,)",
          [MODEL.biasHH],
          "PyTorch keeps this separate from b_ih.",
          false,
          {
            rowLabels: ["b_hh"],
            columnLabels: HIDDEN_NAMES,
            selectedColumn: selectedFeature,
          }
        ),
        matrixCard(
          "Pre-activation",
          "(B,H) = (3,4)",
          record.preActivation,
          "input + memory + b_ih + b_hh",
          true,
          options
        ),
      ].join("");
    }

    return [
      matrixCard(
        "Pre-activation",
        "(B,H) = (3,4)",
        record.preActivation,
        "",
        false,
        options
      ),
      matrixCard(
        "New hidden state h_t",
        "(B,H) = (3,4)",
        record.newHidden,
        "tanh is applied element by element.",
        true,
        options
      ),
    ].join("");
  }

  function renderPostMatrices(info, selectedSentence, selectedFeature) {
    const batchOptions = batchMatrixOptions(
      selectedSentence,
      selectedFeature
    );

    if (info.key === "lastHidden") {
      const selectedLength = WALKTHROUGH.lengths[selectedSentence];
      return [
        matrixCard(
          "Hidden history for selected sentence",
          "(T,H) = (5,4); full history is (3,5,4)",
          WALKTHROUGH.hiddenHistory[selectedSentence],
          "The outlined row at t=length−1 is gathered for classification.",
          false,
          {
            rowLabels: ["t0", "t1", "t2", "t3", "t4"],
            columnLabels: HIDDEN_NAMES,
            selectedRow: selectedLength - 1,
            selectedColumn: selectedFeature,
          }
        ),
        matrixCard(
          "Gathered last-real hidden",
          "(B,H) = (3,4)",
          WALKTHROUGH.lastRealHidden,
          "Indices are lengths−1 = [" +
            WALKTHROUGH.lastIndices.join(", ") +
            "].",
          true,
          batchOptions
        ),
        matrixCard(
          "Hidden state after all padded positions",
          "(B,H) = (3,4); deliberately not used",
          WALKTHROUGH.paddedFinalHidden,
          "For shorter rows this differs from the last-real state: zero PAD input does not freeze recurrence.",
          false,
          batchOptions
        ),
      ].join("");
    }

    if (info.key === "logits") {
      return [
        matrixCard(
          "Last-real hidden",
          "(B,H) = (3,4)",
          WALKTHROUGH.lastRealHidden,
          "",
          false,
          batchOptions
        ),
        matrixCard(
          "Classifier Wᵀ used in row-batch math",
          "(H,C) = (4,2)",
          MODEL.classifierWeightUsed,
          "nn.Linear stores weight with shape (2,4).",
          false,
          {
            rowLabels: HIDDEN_NAMES,
            columnLabels: ["negative", "positive"],
            selectedColumn: WALKTHROUGH.predictions[selectedSentence],
          }
        ),
        matrixCard(
          "Logits",
          "(3,4) @ (4,2) + (2,) = (3,2)",
          WALKTHROUGH.logits,
          "Raw class scores; they are not probabilities yet.",
          true,
          {
            rowLabels: ["S0", "S1", "S2"],
            columnLabels: ["negative", "positive"],
            selectedRow: selectedSentence,
            selectedColumn: WALKTHROUGH.predictions[selectedSentence],
          }
        ),
      ].join("");
    }

    if (info.key === "softmax") {
      return [
        matrixCard(
          "Logits",
          "(B,C) = (3,2)",
          WALKTHROUGH.logits,
          "",
          false,
          {
            rowLabels: ["S0", "S1", "S2"],
            columnLabels: ["negative", "positive"],
            selectedRow: selectedSentence,
          }
        ),
        matrixCard(
          "Softmax probabilities",
          "(B,C) = (3,2)",
          WALKTHROUGH.probabilities,
          "Each row sums to 1. The larger probability is the prediction.",
          true,
          {
            rowLabels: ["S0", "S1", "S2"],
            columnLabels: ["P(negative)", "P(positive)"],
            selectedRow: selectedSentence,
            selectedColumn: WALKTHROUGH.predictions[selectedSentence],
          }
        ),
      ].join("");
    }

    return [
      matrixCard(
        "Targets",
        "(B,) = (3,)",
        [WALKTHROUGH.labels],
        "The correct class for each sentence.",
        false,
        { columnLabels: ["S0", "S1", "S2"] }
      ),
      matrixCard(
        "Predictions",
        "(B,) = (3,)",
        [WALKTHROUGH.predictions],
        "argmax over each probability row.",
        true,
        { columnLabels: ["S0", "S1", "S2"] }
      ),
      matrixCard(
        "Per-sentence cross-entropy",
        "(B,) = (3,)",
        [WALKTHROUGH.rowLosses],
        "For each row: −log(probability assigned to its true class).",
        false,
        { columnLabels: ["S0", "S1", "S2"] }
      ),
    ].join("");
  }

  function calculationRow(label, value, active) {
    return (
      "<dt" +
      (active ? ' class="is-active"' : "") +
      ">" +
      label +
      "</dt><dd" +
      (active ? ' class="is-active"' : "") +
      ">" +
      value +
      "</dd>"
    );
  }

  function detailList(rows) {
    return (
      '<dl class="sentence-rnn__calculation">' +
      rows.join("") +
      "</dl>"
    );
  }

  function rnnScalarInspector(info, selectedSentence, selectedFeature) {
    const record = WALKTHROUGH.records[info.timeStep];
    const x = record.xT[selectedSentence];
    const hPrev = record.hiddenPrevious[selectedSentence];
    const inputTerms = x.map(function (value, embeddingIndex) {
      return (
        formatPrecise(value) +
        "×" +
        formatPrecise(
          MODEL.weightIHUsed[embeddingIndex][selectedFeature]
        )
      );
    });
    const memoryTerms = hPrev.map(function (value, hiddenIndex) {
      return (
        formatPrecise(value) +
        "×" +
        formatPrecise(
          MODEL.weightHHUsed[hiddenIndex][selectedFeature]
        )
      );
    });
    const inputResult =
      record.inputProduct[selectedSentence][selectedFeature];
    const memoryResult =
      record.memoryProduct[selectedSentence][selectedFeature];
    const pre = record.preActivation[selectedSentence][selectedFeature];
    const hidden = record.newHidden[selectedSentence][selectedFeature];

    return (
      '<div class="sentence-rnn__inspector-title">Exact scalar · ' +
      DATASET[selectedSentence].name +
      " · t=" +
      info.timeStep +
      " · " +
      HIDDEN_NAMES[selectedFeature] +
      "</div>" +
      detailList([
        calculationRow(
          "Current x_t",
          "[" + x.map(formatPrecise).join(", ") + "]",
          info.operation === 0
        ),
        calculationRow(
          "Input dot",
          inputTerms.join(" + ") + " = " + formatPrecise(inputResult),
          info.operation === 1
        ),
        calculationRow(
          "Memory dot",
          memoryTerms.join(" + ") + " = " + formatPrecise(memoryResult),
          info.operation === 2
        ),
        calculationRow(
          "Biases",
          formatPrecise(MODEL.biasIH[selectedFeature]) +
            " + " +
            formatPrecise(MODEL.biasHH[selectedFeature]),
          info.operation === 3
        ),
        calculationRow(
          "Pre-activation",
          formatPrecise(inputResult) +
            " + " +
            formatPrecise(memoryResult) +
            " + " +
            formatPrecise(MODEL.biasIH[selectedFeature]) +
            " + " +
            formatPrecise(MODEL.biasHH[selectedFeature]) +
            " = " +
            formatPrecise(pre),
          info.operation === 3
        ),
        calculationRow(
          "New hidden",
          "tanh(" +
            formatPrecise(pre) +
            ") = " +
            formatPrecise(hidden),
          info.operation === 4
        ),
      ])
    );
  }

  function renderExplanation(info, selectedSentence, selectedFeature) {
    const item = DATASET[selectedSentence];

    if (info.key === "text") {
      return {
        plain:
          "<strong>Start with ordinary text.</strong> A batch is simply several training examples processed together. The label is the answer used by the loss; it is not part of the sentence given to the model.",
        inspector:
          '<div class="sentence-rnn__inspector-title">Selected example</div>' +
          detailList([
            calculationRow("Sentence", "“" + escapeHTML(item.text) + "”", true),
            calculationRow(
              "Target",
              item.label + " · " + CLASS_NAMES[item.label],
              false
            ),
            calculationRow("Arithmetic", "None yet", false),
          ]),
      };
    }

    if (info.key === "tokens") {
      return {
        plain:
          "<strong>Tokenization splits one string into an ordered list.</strong> Each word becomes one sequence position. The three lists can have different lengths, so this is still ragged Python data rather than a rectangular tensor.",
        inspector:
          '<div class="sentence-rnn__inspector-title">Selected token list</div>' +
          detailList([
            calculationRow("Tokens", "[" + item.tokens.join(", ") + "]", true),
            calculationRow("Length", String(item.tokens.length), false),
            calculationRow("Shape", "ragged; no tensor shape", false),
          ]),
      };
    }

    if (info.key === "ids") {
      const mappings = item.tokens.map(function (token, index) {
        return token + "→" + item.ids[index];
      });
      return {
        plain:
          "<strong>The vocabulary replaces each word with an integer address.</strong> ID values have no numerical meaning: ID 12 is not “larger” than ID 4. It simply selects another embedding-table row.",
        inspector:
          '<div class="sentence-rnn__inspector-title">Word → ID lookup</div>' +
          detailList([
            calculationRow("Mappings", mappings.join(", "), true),
            calculationRow("IDs", "[" + item.ids.join(", ") + "]", false),
            calculationRow("Length", String(item.ids.length), false),
          ]),
      };
    }

    if (info.key === "padding") {
      return {
        plain:
          "<strong>Padding makes every row equally wide.</strong> The longest sentence has five words, so shorter rows receive PAD ID 0 on the right. The separate lengths tensor remembers which positions are real.",
        inspector:
          '<div class="sentence-rnn__inspector-title">Selected padded row</div>' +
          detailList([
            calculationRow(
              "Before",
              "[" + item.ids.join(", ") + "]",
              false
            ),
            calculationRow(
              "After",
              "[" + WALKTHROUGH.paddedIDs[selectedSentence].join(", ") + "]",
              true
            ),
            calculationRow(
              "Added PADs",
              String(T - item.ids.length),
              false
            ),
            calculationRow("Result shape", "(B,T) = (3,5)", false),
          ]),
      };
    }

    if (info.key === "embedding") {
      const firstID = WALKTHROUGH.paddedIDs[selectedSentence][0];
      return {
        plain:
          "<strong>An embedding lookup changes each ID into three trainable numbers.</strong> It is row selection, not multiplication by the ID. The complete batch grows from (3,5) integer addresses to (3,5,3) floating-point values.",
        inspector:
          '<div class="sentence-rnn__inspector-title">One lookup · ' +
          item.tokens[0] +
          "</div>" +
          detailList([
            calculationRow("Token ID", String(firstID), false),
            calculationRow(
              "Selected row",
              "embedding[" + firstID + "]",
              false
            ),
            calculationRow(
              "Vector",
              "[" + MODEL.embedding[firstID].map(formatPrecise).join(", ") + "]",
              true
            ),
            calculationRow("Batch result", "(B,T,E) = (3,5,3)", false),
          ]),
      };
    }

    if (info.kind === "rnn") {
      const isPadding =
        info.timeStep >= WALKTHROUGH.lengths[selectedSentence];
      const paddingWarning = isPadding
        ? '<p class="sentence-rnn__warning"><strong>This is PAD for the selected row.</strong> Its embedding is zero, but the recurrent path and biases still update memory. This state will not be selected for classification.</p>'
        : "";
      const explanations = [
        "<strong>Select one position from every sentence.</strong> x[:,t,:] removes the time axis for this operation: (3,5,3) → (3,3). The RNN processes all three rows together.",
        "<strong>The input path translates the current word vectors into hidden space.</strong> (3,3) @ (3,4) produces one four-feature contribution per sentence.",
        "<strong>The memory path transforms the previous hidden state.</strong> This is what lets information from earlier words affect the current step. The same matrix is reused at every timestep.",
        "<strong>Add the two paths and both PyTorch RNN biases.</strong> All four terms produce or broadcast to shape (3,4), so the sum keeps shape (3,4).",
        "<strong>tanh creates the new hidden state.</strong> It acts element by element and keeps the values between −1 and 1. This h_t becomes h_prev for the next timestep.",
      ];
      return {
        plain: explanations[info.operation] + paddingWarning,
        inspector: rnnScalarInspector(
          info,
          selectedSentence,
          selectedFeature
        ),
      };
    }

    if (info.key === "lastHidden") {
      const length = WALKTHROUGH.lengths[selectedSentence];
      const chosen = WALKTHROUGH.lastRealHidden[selectedSentence];
      const padded = WALKTHROUGH.paddedFinalHidden[selectedSentence];
      return {
        plain:
          "<strong>Each sentence must be summarized at its own real endpoint.</strong> The required state is gathered from hidden_history[row, length−1, :]. Raw nn.RNN still computes padded positions, and a zero PAD embedding does not freeze its recurrent memory.",
        inspector:
          '<div class="sentence-rnn__inspector-title">Gather one row</div>' +
          detailList([
            calculationRow("Length", String(length), false),
            calculationRow("Last-real index", length + "−1 = " + (length - 1), true),
            calculationRow(
              "Chosen hidden",
              "[" + chosen.map(formatPrecise).join(", ") + "]",
              true
            ),
            calculationRow(
              "After all PADs",
              "[" + padded.map(formatPrecise).join(", ") + "]",
              false
            ),
            calculationRow("Batch result", "(B,H) = (3,4)", false),
          ]),
      };
    }

    if (info.key === "logits") {
      const hidden = WALKTHROUGH.lastRealHidden[selectedSentence];
      const rows = Array.from({ length: C }, function (_, classIndex) {
        const terms = hidden.map(function (value, hiddenIndex) {
          return (
            formatPrecise(value) +
            "×" +
            formatPrecise(
              MODEL.classifierWeightUsed[hiddenIndex][classIndex]
            )
          );
        });
        return calculationRow(
          CLASS_NAMES[classIndex] + " logit",
          terms.join(" + ") +
            " + " +
            formatPrecise(MODEL.classifierBias[classIndex]) +
            " = " +
            formatPrecise(
              WALKTHROUGH.logits[selectedSentence][classIndex]
            ),
          WALKTHROUGH.predictions[selectedSentence] === classIndex
        );
      });
      return {
        plain:
          "<strong>The linear classifier converts four hidden features into two class scores.</strong> A logit is an unrestricted score, not a probability. The same classifier weights are used for every sentence.",
        inspector:
          '<div class="sentence-rnn__inspector-title">Classifier dot products · ' +
          item.name +
          "</div>" +
          detailList(rows),
      };
    }

    if (info.key === "softmax") {
      const logits = WALKTHROUGH.logits[selectedSentence];
      const maximum = Math.max.apply(null, logits);
      const exps = logits.map(function (value) {
        return Math.exp(value - maximum);
      });
      const denominator = exps[0] + exps[1];
      return {
        plain:
          "<strong>Softmax turns the two logits into probabilities that sum to one.</strong> Subtracting the row maximum before exponentiation changes neither probability, but makes the calculation numerically stable.",
        inspector:
          '<div class="sentence-rnn__inspector-title">Stable softmax · ' +
          item.name +
          "</div>" +
          detailList([
            calculationRow(
              "Logits",
              "[" + logits.map(formatPrecise).join(", ") + "]",
              false
            ),
            calculationRow("Row max", formatPrecise(maximum), false),
            calculationRow(
              "Exponentials",
              "[" + exps.map(formatPrecise).join(", ") + "]",
              false
            ),
            calculationRow("Denominator", formatPrecise(denominator), false),
            calculationRow(
              "Probabilities",
              "[" +
                WALKTHROUGH.probabilities[selectedSentence]
                  .map(formatPrecise)
                  .join(", ") +
                "]",
              true
            ),
          ]),
      };
    }

    const targetProbability =
      WALKTHROUGH.probabilities[selectedSentence][item.label];
    return {
      plain:
        "<strong>Cross-entropy measures confidence in the correct labels; accuracy counts correct decisions.</strong> The batch loss is the mean of three row losses. During training, this scalar loss would drive backpropagation through every earlier stage.",
      inspector:
        '<div class="sentence-rnn__inspector-title">Loss and decision · ' +
        item.name +
        "</div>" +
        detailList([
          calculationRow("Target", String(item.label), false),
          calculationRow(
            "Prediction",
            String(WALKTHROUGH.predictions[selectedSentence]),
            true
          ),
          calculationRow(
            "Target probability",
            formatPrecise(targetProbability),
            false
          ),
          calculationRow(
            "Row loss",
            "−log(" +
              formatPrecise(targetProbability) +
              ") = " +
              formatPrecise(WALKTHROUGH.rowLosses[selectedSentence]),
            false
          ),
          calculationRow(
            "Batch mean loss",
            formatPrecise(WALKTHROUGH.loss),
            false
          ),
        ]),
    };
  }

  function renderMetrics() {
    return (
      '<div class="sentence-rnn__metric-grid">' +
      '<div><strong>Labels · (3,)</strong><code>[' +
      WALKTHROUGH.labels.join(", ") +
      "]</code></div>" +
      '<div><strong>Predictions · (3,)</strong><code>[' +
      WALKTHROUGH.predictions.join(", ") +
      "]</code></div>" +
      '<div><strong>Batch accuracy · scalar</strong><code>' +
      WALKTHROUGH.correct +
      "/" +
      B +
      " = " +
      (100 * WALKTHROUGH.accuracy).toFixed(1) +
      "%</code></div>" +
      '<div><strong>Mean cross-entropy · scalar</strong><code>' +
      formatPrecise(WALKTHROUGH.loss) +
      "</code></div>" +
      "</div>"
    );
  }

  function stageStatus(info, position) {
    const prefix =
      "Operation " + (position + 1) + " of " + (FINAL_POSITION + 1) + " · ";
    if (info.kind === "rnn") {
      return (
        prefix +
        "RNN · t=" +
        info.timeStep +
        " · " +
        CELL_OPERATIONS[info.operation]
      );
    }
    return prefix + info.title;
  }

  function initialize(container, instanceIndex) {
    if (container.dataset.sentenceRnnReady === "true") return;
    container.dataset.sentenceRnnReady = "true";

    const shell = container.closest("[data-sentence-rnn-shell]");
    const openFullscreenButton = container.querySelector(
      "[data-sentence-rnn-fullscreen-open]"
    );
    const closeFullscreenButton = container.querySelector(
      "[data-sentence-rnn-fullscreen-close]"
    );
    const titleElement = container.querySelector(".sentence-rnn__title");
    const statusElement = container.querySelector(
      "[data-sentence-rnn-status]"
    );
    const progressElement = container.querySelector(
      "[data-sentence-rnn-progress]"
    );
    const stageNavElement = container.querySelector(
      "[data-sentence-rnn-stage-nav]"
    );
    const timestepNavElement = container.querySelector(
      "[data-sentence-rnn-timestep-nav]"
    );
    const operationNavElement = container.querySelector(
      "[data-sentence-rnn-operation-nav]"
    );
    const sentencePickerElement = container.querySelector(
      "[data-sentence-rnn-sentence-picker]"
    );
    const featurePickerElement = container.querySelector(
      "[data-sentence-rnn-feature-picker]"
    );
    const batchElement = container.querySelector("[data-sentence-rnn-batch]");
    const matricesElement = container.querySelector(
      "[data-sentence-rnn-matrices]"
    );
    const metricsElement = container.querySelector(
      "[data-sentence-rnn-metrics]"
    );
    const explanationElement = container.querySelector(
      "[data-sentence-rnn-explanation]"
    );
    const inspectorElement = container.querySelector(
      "[data-sentence-rnn-inspector]"
    );
    const scrubber = container.querySelector("[data-sentence-rnn-scrubber]");
    const playButton = container.querySelector(
      '[data-sentence-rnn-action="play"]'
    );
    const previousButton = container.querySelector(
      '[data-sentence-rnn-action="previous"]'
    );
    const nextButton = container.querySelector(
      '[data-sentence-rnn-action="next"]'
    );
    const resetButton = container.querySelector(
      '[data-sentence-rnn-action="reset"]'
    );
    const speedControl = container.querySelector("[data-sentence-rnn-speed]");
    const tabs = Array.from(
      container.querySelectorAll("[data-sentence-rnn-tab]")
    );
    const panels = Array.from(
      container.querySelectorAll("[data-sentence-rnn-panel]")
    );

    const titleID = "sentence-rnn-title-" + instanceIndex;
    titleElement.id = titleID;

    tabs.forEach(function (tab) {
      const name = tab.dataset.sentenceRnnTab;
      const panel = container.querySelector(
        '[data-sentence-rnn-panel="' + name + '"]'
      );
      const tabID = "sentence-rnn-tab-" + instanceIndex + "-" + name;
      const panelID = "sentence-rnn-panel-" + instanceIndex + "-" + name;
      tab.id = tabID;
      tab.setAttribute("aria-controls", panelID);
      panel.id = panelID;
      panel.setAttribute("aria-labelledby", tabID);
    });

    let position = 0;
    let selectedSentence = 0;
    let selectedFeature = 0;
    let activeTab = "journey";
    let timer = null;
    let isFullscreen = false;
    let restoreFocusElement = null;
    const fullscreenController = { close: closeFullscreen };

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
      openFullscreenButton.setAttribute("aria-expanded", "true");
      closeFullscreenButton.hidden = false;
      document.documentElement.classList.add("sentence-rnn-modal-open");
      document.body.classList.add("sentence-rnn-modal-open");
      document.addEventListener("keydown", handleFullscreenKeydown);
      container.scrollTop = 0;

      window.requestAnimationFrame(function () {
        focusWithoutScroll(closeFullscreenButton);
      });
    }

    function closeFullscreen(options) {
      if (!isFullscreen) return;

      const shouldRestoreFocus =
        !options || options.restoreFocus !== false;
      isFullscreen = false;
      stop();
      shell.classList.remove("is-fullscreen");
      container.classList.remove("is-fullscreen");
      container.removeAttribute("role");
      container.removeAttribute("aria-modal");
      container.removeAttribute("aria-labelledby");
      openFullscreenButton.setAttribute("aria-expanded", "false");
      closeFullscreenButton.hidden = true;
      document.removeEventListener("keydown", handleFullscreenKeydown);

      if (activeFullscreenController === fullscreenController) {
        activeFullscreenController = null;
        document.documentElement.classList.remove(
          "sentence-rnn-modal-open"
        );
        document.body.classList.remove("sentence-rnn-modal-open");
      }

      if (
        shouldRestoreFocus &&
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
        const active = tab.dataset.sentenceRnnTab === activeTab;
        tab.classList.toggle("is-current", active);
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
        if (active && focus) tab.focus();
      });
      panels.forEach(function (panel) {
        panel.hidden = panel.dataset.sentenceRnnPanel !== activeTab;
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
        selectedFeature
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
      scrubber.setAttribute(
        "aria-valuetext",
        stageStatus(info, position)
      );

      stageNavElement.innerHTML = renderStageNavigation(position);
      const currentStageButton = stageNavElement.querySelector(
        '[aria-current="step"]'
      );
      if (currentStageButton) {
        stageNavElement.scrollLeft = Math.max(
          0,
          currentStageButton.offsetLeft -
            (stageNavElement.clientWidth - currentStageButton.offsetWidth) / 2
        );
      }
      timestepNavElement.innerHTML = renderTimestepNavigation(info);
      operationNavElement.innerHTML = renderOperationNavigation(info);
      sentencePickerElement.innerHTML =
        renderSentencePicker(selectedSentence);
      featurePickerElement.innerHTML =
        renderFeaturePicker(selectedFeature);
      batchElement.innerHTML = renderBatch(info, selectedSentence);

      if (info.kind === "pre") {
        matricesElement.innerHTML = renderPreMatrices(
          info,
          selectedSentence
        );
      } else if (info.kind === "rnn") {
        matricesElement.innerHTML = renderRNNMatrices(
          info,
          selectedSentence,
          selectedFeature
        );
      } else {
        matricesElement.innerHTML = renderPostMatrices(
          info,
          selectedSentence,
          selectedFeature
        );
      }

      explanationElement.innerHTML = explanation.plain;
      inspectorElement.innerHTML = explanation.inspector;
      metricsElement.hidden = info.key !== "metrics";
      metricsElement.innerHTML =
        info.key === "metrics" ? renderMetrics() : "";

      if (position === FINAL_POSITION) stop();
    }

    function jumpTo(nextPosition) {
      stop();
      position = Math.max(0, Math.min(FINAL_POSITION, nextPosition));
      render();
    }

    function focusWithoutScroll(element) {
      if (!element) return;
      try {
        element.focus({ preventScroll: true });
      } catch (_error) {
        element.focus();
      }
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

    resetButton.addEventListener("click", function () {
      jumpTo(0);
    });
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
    openFullscreenButton.addEventListener("click", openFullscreen);
    closeFullscreenButton.addEventListener("click", function () {
      closeFullscreen();
    });
    shell.addEventListener("click", function (event) {
      if (isFullscreen && event.target === shell) {
        closeFullscreen();
      }
    });

    container.addEventListener("click", function (event) {
      const tab = event.target.closest("[data-sentence-rnn-tab]");
      if (tab && container.contains(tab)) {
        activateTab(tab.dataset.sentenceRnnTab, false);
        return;
      }

      const jump = event.target.closest("[data-sentence-rnn-jump]");
      if (jump && container.contains(jump)) {
        const jumpValue = jump.dataset.sentenceRnnJump;
        const navigationHost = jump.closest(
          "[data-sentence-rnn-stage-nav], " +
            "[data-sentence-rnn-timestep-nav], " +
            "[data-sentence-rnn-operation-nav]"
        );
        const hostAttribute = navigationHost
          ? Array.from(navigationHost.attributes).find(function (attribute) {
              return attribute.name.startsWith("data-sentence-rnn-");
            }).name
          : null;
        jumpTo(Number(jumpValue));
        const replacementHost = hostAttribute
          ? container.querySelector("[" + hostAttribute + "]")
          : container;
        focusWithoutScroll(
          replacementHost.querySelector(
            '[data-sentence-rnn-jump="' + jumpValue + '"]'
          )
        );
        return;
      }

      const sentence = event.target.closest(
        "[data-sentence-rnn-select-sentence]"
      );
      if (sentence && container.contains(sentence)) {
        const sentenceValue = sentence.dataset.sentenceRnnSelectSentence;
        const cameFromPicker = Boolean(
          sentence.closest("[data-sentence-rnn-sentence-picker]")
        );
        selectedSentence = Number(sentenceValue);
        stop();
        render();
        const replacementRoot = cameFromPicker
          ? sentencePickerElement
          : batchElement;
        focusWithoutScroll(
          replacementRoot.querySelector(
            '[data-sentence-rnn-select-sentence="' +
              sentenceValue +
              '"]'
          )
        );
        return;
      }

      const feature = event.target.closest(
        "[data-sentence-rnn-select-feature]"
      );
      if (feature && container.contains(feature)) {
        const featureValue = feature.dataset.sentenceRnnSelectFeature;
        selectedFeature = Number(featureValue);
        stop();
        render();
        focusWithoutScroll(
          featurePickerElement.querySelector(
            '[data-sentence-rnn-select-feature="' +
              featureValue +
              '"]'
          )
        );
      }
    });

    container.addEventListener("keydown", function (event) {
      const tab = event.target.closest("[data-sentence-rnn-tab]");
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
        activateTab(tabs[nextIndex].dataset.sentenceRnnTab, true);
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

  // Read-only values for lightweight consistency tests. The UI never reads
  // this hook, so removing it would not affect production behavior.
  window.__sentenceRNNTest = deepFreeze({
    dimensions: { B: B, T: T, E: E, H: H, C: C },
    vocabulary: VOCABULARY.slice(),
    dataset: DATASET.map(function (item) {
      return {
        name: item.name,
        text: item.text,
        tokens: item.tokens.slice(),
        ids: item.ids.slice(),
        label: item.label,
      };
    }),
    model: JSON.parse(JSON.stringify(MODEL)),
    walkthrough: JSON.parse(JSON.stringify(WALKTHROUGH)),
    positions: {
      recurrenceStart: RECURRENCE_START,
      lastHidden: LAST_HIDDEN_POSITION,
      logits: LOGITS_POSITION,
      softmax: SOFTMAX_POSITION,
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
      .querySelectorAll("[data-sentence-rnn]")
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
