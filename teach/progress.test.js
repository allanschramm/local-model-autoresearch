const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");

function loadProgress(initial = {}) {
  const saved = new Map(Object.entries(initial));
  const context = {
    console,
    localStorage: {
      getItem(key) {
        return saved.has(key) ? saved.get(key) : null;
      },
      setItem(key, value) {
        saved.set(key, String(value));
      },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(
    fs.readFileSync(path.join(__dirname, "assets", "progress.js"), "utf8"),
    context,
  );
  return context.TeachProgress;
}

test("published journey includes Module 0, Week 1, and Week 2 Days 1–2", () => {
  const progress = loadProgress();

  assert.deepEqual(
    Array.from(progress.LESSON_ORDER, (lesson) => lesson.id),
    ["s0d1", "s0d2", "s1d1", "s1d2", "s1d3", "s1d4", "s2d1", "s2d2"],
  );
  assert.equal(
    progress.getNextLesson([
      "s0d1",
      "s0d2",
      "s1d1",
      "s1d2",
      "s1d3",
      "s1d4",
      "s2d1",
      "s2d2",
    ]),
    null,
  );
  assert.equal(progress.isPublishedLesson("s2d1"), true);
  assert.equal(progress.isPublishedLesson("s2d2"), true);
});

test("lesson readiness requires quizzes and either practice route", () => {
  const progress = loadProgress({
    teach_quiz_pass_v1: JSON.stringify(["s0d2-q1", "s0d2-q2"]),
  });

  assert.equal(progress.isLessonReady("s0d2"), false);
  progress.markPractice("s0d2", "simulated");
  assert.equal(progress.isLessonReady("s0d2"), true);
  assert.equal(progress.getPracticeMode("s0d2"), "simulated");

  progress.markPractice("s0d2", "real");
  assert.equal(progress.getPracticeMode("s0d2"), "real");
});

test("Week 2 Day 1 readiness needs all four quizzes plus practice", () => {
  const progress = loadProgress({
    teach_quiz_pass_v1: JSON.stringify(["s2d1-q1", "s2d1-q2", "s2d1-q3"]),
  });

  assert.equal(progress.isLessonReady("s2d1"), false);
  progress.markQuizPassed("s2d1-q4");
  progress.markPractice("s2d1", "simulated");
  assert.equal(progress.isLessonReady("s2d1"), true);
});

test("Week 2 Day 2 readiness needs all four quizzes plus practice", () => {
  const progress = loadProgress({
    teach_quiz_pass_v1: JSON.stringify(["s2d2-q1", "s2d2-q2", "s2d2-q3"]),
  });

  assert.equal(progress.isLessonReady("s2d2"), false);
  progress.markQuizPassed("s2d2-q4");
  progress.markPractice("s2d2", "simulated");
  assert.equal(progress.isLessonReady("s2d2"), true);
});

test("unpublished Week 2 drafts stay out of LESSON_ORDER", () => {
  const progress = loadProgress();
  assert.equal(progress.isPublishedLesson("s2d3"), false);
  assert.equal(progress.isPublishedLesson("s2d4"), false);
});
