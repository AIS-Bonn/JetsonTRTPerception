#ifndef DATASETS_H
#define DATASETS_H

struct ADE20K_INDOOR{
  static constexpr int background = 0,
  wall = 1,
  floor = 2,
  ceiling = 3,
  window = 4,
  door = 5,
  column = 6,
  stairs = 7,
  table = 8,
  chair = 9,
  seat = 10,
  cabinet = 11,
  shelf = 12,
  lamp = 13,
  person = 14,
  animal = 15,
  vehicle = 16,
  bike = 17,
  poster = 18,
  box = 19,
  book = 20,
  toy = 21,
  fridge = 22,
  dishwasher = 23,
  oven = 24,
  trashbin = 25,
  computer = 26,
  tv = 27,
  screen = 28,
  glass = 29,
  bottle = 30,
  food = 31,
  NUM_CLASSES = 32;
};

#endif // DATASETS_H
