---
tags: [other-languages, typescript, nodejs]
summary: Node/TypeScript toolchain — fnm, npm/pnpm/Yarn, corepack, project setup, hello world, modern JS and the TS handbook.
---
# ___Typescript___
***

# Node and TypeScript Toolchain
## Links
- [Node.js](https://nodejs.org/)
- [TypeScript](https://www.typescriptlang.org/docs/)
- [TypeScript handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [pnpm](https://pnpm.io/)
- [fnm](https://github.com/Schniz/fnm)
- [Yarn](https://yarnpkg.com/)
- [Deno](https://docs.deno.com/)
## Concepts

| Tool | Purpose | Installation |
| --- | --- | --- |
| `node` | JS runtime (V8 + libuv); supports TS type stripping in modern versions | Managed by `fnm` |
| `npm` | Default package manager, bundled with Node | Comes with Node |
| `npx` | Package binary runner, bundled with npm | Comes with npm |
| `pnpm` | Alternative package manager; shared content-addressable store, saves disk space | Standalone, `~/.local/share/pnpm/` |
| `yarn` | Alternative package manager; Classic (1.x) and Modern (2+); useful for existing Yarn projects | Corepack or standalone |
| `fnm` | Fast Node version manager (Rust); switches Node versions on `PATH` | `~/.fnm/` |
| `corepack` | Selects project-pinned pnpm/Yarn versions; no longer bundled from Node 25 | Bundled with Node ≤24 or installed separately |
| `tsc` | TypeScript compiler and type checker; provided by npm package `typescript` | Project `node_modules/.bin/` |
| `tsx` | Execute TS through Node, with watch mode; no type checking | npm/pnpm package |
| `deno` | Independent JS/TS runtime with built-in type checking, formatter, linter, REPL | Standalone or `npm install -g deno` |
| `nodejs` | Debian/Ubuntu package name; may also be an executable alias | apt, not needed with fnm |

**Relationship:** `fnm` manages Node versions; npm/pnpm/Yarn manage project dependencies; `tsc` checks types; Node/tsx/Deno execute code. npm, pnpm, and Yarn generally use the same npm registry.
## Environment setup
- Target: **fnm + Node/npm/npx + standalone pnpm**. Yarn, Corepack, and Deno are optional.
- **Remove conflicting apt-installed Node (if present):**
  ```sh
  sudo apt purge -y nodejs npm libnode-dev
  sudo apt autoremove -y
  # Review any proposed autoremove before running it.
  grep -n 'prefix' ~/.npmrc 2>/dev/null
  ```
  Remove stale npm prefixes pointing to `/usr` or `/usr/local`.
- **Install fnm:**
  ```sh
  curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "$HOME/.fnm" --skip-shell
  ```
  Add to `~/.bashrc`:
  ```sh
  export PATH="$HOME/.fnm:$PATH"
  eval "$(fnm env --use-on-cd --shell bash)"
  ```
- **Install Node:**
  ```sh
  fnm install --lts
  fnm use --lts
  fnm default 24  # or whatever --lts resolved to

  node --version
  npm --version
  npx --version
  ```
  Optional npm upgrade (requires a compatible Node version):
  ```bash
  npm install -g npm@latest
  ```
- **Install pnpm standalone:**
  ```sh
  curl -fsSL https://get.pnpm.io/install.sh | sh -
  source ~/.bashrc
  pnpm --version

  pnpm setup
  source ~/.bashrc
  ```
  The installer configures `PNPM_HOME`, typically `~/.local/share/pnpm`. Binaries provided: `pnpm` (main), `pn` (alias), `pnpx` (legacy = `pnpm dlx`), `pnx` (run a package binary).
## Package manager comparison

| Task                 | npm                  | pnpm              | Yarn              |
| -------------------- | -------------------- | ----------------- | ----------------- |
| Initialize           | `npm init -y`        | `pnpm init`       | `yarn init`       |
| Install dependencies | `npm install`        | `pnpm install`    | `yarn install`    |
| Add package          | `npm install foo`    | `pnpm add foo`    | `yarn add foo`    |
| Add dev package      | `npm install -D foo` | `pnpm add -D foo` | `yarn add -D foo` |
| Execute local binary | `npx tsc`            | `pnpm exec tsc`   | `yarn tsc`        |
| Run script           | `npm run dev`        | `pnpm dev`        | `yarn dev`        |
| Lockfile             | `package-lock.json`  | `pnpm-lock.yaml`  | `yarn.lock`       |

Use **pnpm** for new projects; follow the existing package manager in other repositories. Avoid mixing lockfiles.
## Routine maintenance
```sh
# Node
fnm list                          # installed versions
fnm ls-remote --lts               # available LTS releases
fnm use 22                        # switch this shell only
fnm default 24                    # switch the default for new shells
fnm exec --using=22 -- node app.js # one-off run on another version
fnm install --lts && fnm default <new-version>
npm install -g npm@latest        # per Node version — re-run after every upgrade

# pnpm
pnpm self-update
pnpm store path
pnpm store prune              # Optional cache cleanup

# Project
pnpm outdated
pnpm up
pnpm up --latest              # Includes possible major upgrades
```
## Verification
```sh
type -a node npm npx pnpm yarn fnm  # each must resolve to exactly ONE path
node -p 'process.execPath'
npm config get prefix  # must be inside the fnm version dir, NOT /usr
pnpm store path  # must match the current major (e.g. .../store/v11)
```
Multiple PATH matches are not necessarily wrong; inspect which executable resolves first. Expected shape:
```sh
node  -> /run/user/<uid>/fnm_multishells/<pid>_<ts>/bin/node
npm   -> /run/user/<uid>/fnm_multishells/<pid>_<ts>/bin/npm
pnpm  -> /home/<user>/.local/share/pnpm/bin/pnpm
```
## Full tear down
  ```sh
  rm -rf ~/.fnm ~/.local/share/fnm
  rm -rf ~/.local/share/pnpm ~/.cache/pnpm
  rm -rf ~/.npm ~/.npmrc
  # then strip the fnm / PNPM_HOME blocks from ~/.bashrc
  ```
***

## Hello World
- **npm commands**
  - `npm install` foo	Add to runtime dependencies
  - `npm install -D foo` Add to development dependencies
  - `npm install -g foo` Install globally
  - `npm install` Install dependencies declared by the project
  - `npm ci` Clean, reproducible installation using the lockfile
  - `npm install --omit=dev` Install without development dependencies
  - `npm uninstall foo` Remove a package
- **Initialize TypeScript project**. No npm project required for simple scripts.
  ```sh
  pnpm init  # Create package.json
  pnpm add -D typescript  # Install tsc locally
  pnpm exec tsc --init  # Create tsconfig.json

  pnpm approve-builds  # [Optional] Need to approve `esbuild` if need tsx
  pnpm add -D tsx  # [Optional] execution tool
  ```
  `hello_world.ts`
  ```ts
  const hello = "Hello";
  const world = "World";
  console.log(hello + " " + world);
  ```
  Run
  ```sh
  node hello_world.ts  # Native TS type stripping
  pnpm exec tsx hello_world.ts  # Alternative runner
  pnpm exec tsx watch hello_world.ts  # with automatic restart on file changes
  pnpm exec tsc --noEmit  # Type-check project
  pnpm exec tsc  # Compile to JS
  ```
- **Deno alternative:**
  ```sh
  # pnpm add -g deno  # [Optional] Another toolchain
  curl -fsSL https://deno.land/install.sh | sh
  ```
  Fix for `deno run` auto-complete
  ```sh
  deno completions bash > ~/.deno-completions.bash
  sed -i 's/\\n/\ /g' ~/.deno-completions.bash
  source ~/.deno-completions.bash
  ```
  **Run**
  ```bash
  deno run hello_world.ts
  deno check hello_world.ts
  deno repl  # Interactive TS console
  ```
***

# JavaScript Examples
## JavaScript Essentials
- **Objects and arrays** JavaScript objects are collections of values wrapped up with named keys
  ```ts
  const userAccount = {
    name: "Kieron",
    id: 0,
  };

  const pie = {
    type: "Apple",
  };

  // You can combine these to make larger, more complex data-models.
  const purchaseOrder = {
    owner: userAccount,
    item: pie,
  };

  // Values can be accessed via the ".", so to get a username for a purchase order:
  console.log(purchaseOrder.item.type);

  // TypeScript also offers these same features to arrays.
  const allOrders = [purchaseOrder];
  const firstOrder = allOrders[0];
  console.log(firstOrder.item.type);

  // An alternative way to get an object is via pop-ing the array to remove objects.
  const poppedFirstOrder = allOrders.pop();

  // Creates a type based on the shape of a purchase order:
  type PurchaseOrder = typeof purchaseOrder;
  // Creates a readonly array of purchase orders
  const readonlyOrders: readonly PurchaseOrder[] = [purchaseOrder];
  // Now if you try to pop from the readonlyOrders, TypeScript will raise an error.
  readonlyOrders.pop();
  ```
- **Functions**
  ```ts
  function addOldSchool(x, y) {
    return x + y;
  }
  // You can move the name of the function to a variable name
  const goo = function (x, y) { return x+y; };
  // Or
  const foo = (x, y) => { return x+y; };

  // TypeScript provides additional syntax which adds to a function definition and offers hints on what types are expected by this function.
  const add1 = (x: any, y: any) => {
    return x + y;
  };
  add1("Hello", 23);  // "hello23"

  // x and y to only be numbers.
  const add2 = (x: number, y: number) => {
    return x + y;
  };
  add2(16, 23);
  add2("Hello", 23);  // deno check throws error. Argument of type 'string' is not assignable to parameter of type 'number'.

  // return type should also be a number.
  const add4 = (x: number, y: number): number => {
    return x + y;
  };
  // This function fails. [ERROR]: Type 'number' is not assignable to type 'string'
  const add3 = (x: number, y: number): string => {
    return x + y;
  };
  ```
  In JavaScript/TypeScript arrow functions, parentheses are optional when there is exactly one parameter and it is a simple identifier.
    - Zero parameter: `() => 42`
    - One parameter: `x => x * 2` or `(x) => x * 2`
    - One parameter with typed: `x: number => x * 2` or `(x: number) => x * 2`
    - Two parameters: `x, y => x + y` or `(x, y) => x + y`
    - Default value: `(x = 1) => x + 1`
- **Code flow**
  ```ts
  const users = [{ name: "Ahmed" }, { name: "Gemma" }, { name: "Jon" }];
  // Find an element
  const jon = users.find((u) => u.name === "Jon");

  // Failed will create the type '{ name: string } | undefined'
  const jon = users.find((u) => u.name === "jon");

  if (jon) {
    jon;
  } else {
    jon;
  }

  // We can use the JavaScript 'typeof x === y' syntax to check for the type of the elements
  const identifiers = ["Hello", "World", 24, 19];
  identifiers.find((uu) => typeof uu === 'number')  // 24
  identifiers.findLast((uu) => typeof uu === 'number')  // 19
  ```
## Functions with JavaScript
- **Generic functions** Generics provide a way to use Types as variables in other types. Meta
  ```ts
  // Generics provide a way to use Types as variables in other types. Meta.
  function wrapInArray<Type>(input: Type): Type[] {
    return [input];
  }

  const stringArray = wrapInArray("hello generics");
  const numberArray = wrapInArray(123);
  // You can also skip the generic inference by adding the type yourself also:
  const stringArray2 = wrapInArray<string>("");

  // Use the type has to exte/nd a particular type to only allow a subset of types.
  interface Drawable {
    draw: () => void;
  }

  // This function takes a set of objects which have a function for drawing to the screen
  function renderToScreen<Type extends Drawable>(input: Type[]) {
    input.forEach((i) => i.draw());
  }

  const objectsWithDraw = [{ draw: () => {} }, { draw: () => {} }];
  renderToScreen(objectsWithDraw);

  // It will fail if draw is missing:
  renderToScreen([{}, { draw: () => {} }]);

  // An example of a caching function that lets you have different sets of input types and caches.
  interface CacheHost {
    save: (a: any) => void;
  }

  function addObjectToCache<Type, Cache extends CacheHost>(obj: Type, cache: Cache): Cache {
    cache.save(obj);
    return cache;
  }

  // This is the same as above, but with an extra parameter.
  interface CacheHostGeneric<ContentType> {
    save: (a: ContentType) => void;
  }

  // Now when the CacheHostGeneric is used, you need to tell it what ContentType is.
  function addTypedObjectToCache<Type, Cache extends CacheHostGeneric<Type>>(obj: Type, cache: Cache): Cache {
    cache.save(obj);
    return cache;
  }
  ```
- **Typing Functions** TypeScript's inference can get you very far, but there are lots of extra ways to provide a richer way to document the shape of your functions.
  ```ts
  // Optional params
  let i = 0;
  const incrementIndex = (value?: number) => {
    i += value === undefined ? 1 : value;
  };

  // This function can be called like:
  incrementIndex();
  incrementIndex(0);
  incrementIndex(3);

  // You can type parameters as functions, which provides type inference when you write the functions.
  const callbackWithIndex = (callback: (i: number) => void) => {
    callback(i);
  };

  // Using a type alias will let you name the function param.
  type NumberCallback = (i: number) => void;
  const callbackWithIndex2 = (callback: NumberCallback) => {
    callback(i);
  };

  // These can be called like:
  callbackWithIndex(index => {
    console.log(index);
  });

  // TypeScript inference can work when passing a function as an instance reference too
  const numberToString = (n: number) => {
    return n.toString();
  };

  // This can be used in a function like map on an array to convert all numbers into a string.
  const stringedNumbers = [1, 4, 6, 10].map(i => numberToString(i));

  // We can use shorthand to have the function passed directly and get the same results with more focused code:
  const stringedNumbersTerse = [1, 4, 6, 10].map(numberToString);

  // The following type declares that this function is OK to use any object so long as it includes the property name:
  interface AnyObjectButMustHaveName {
    name: string;
    [key: string]: any;
  }
  const printFormattedName = (input: AnyObjectButMustHaveName) => {};
  printFormattedName({ name: "joey" });
  printFormattedName({ name: "joey", age: 23 });

  // Using union types in your parameters makes sense if there are only one or two changes and documentation does not need to change.
  const boolOrNumberFunction = (input: boolean | number) => {};
  boolOrNumberFunction(true);
  boolOrNumberFunction(23);

  // Function overloads on the other hand offer a much richer syntax for the parameters and return types.
  interface BoolOrNumberOrStringFunction {
    /** Takes a bool, returns a bool */
    (input: boolean): boolean;
    /** Takes a number, returns a number */
    (input: number): number;
    /** Takes a string, returns a bool */
    (input: string): boolean;
  }

  // Declare, it allows you to tell TypeScript something exists even if it doesn't exist in the runtime in this file.
  declare const boolOrNumberOrStringFunction: BoolOrNumberOrStringFunction;

  const boolValue = boolOrNumberOrStringFunction(true);
  const numberValue = boolOrNumberOrStringFunction(12);
  const boolValue2 = boolOrNumberOrStringFunction("string");
  ```
- **Function Chaining** Function chaining APIs can make your code focused with less intermediary values and easier to read because of their nesting qualities.
  ```ts
  import $ from "jquery";

  // Here's an example use of the jQuery API:
  $("#navigation").css("background", "red").height(300).fadeIn(200);

  // An example API which creates a chaining API. The key is to have an outer function which
  // keeps track of internal state, and an object which exposes the API that is always returned.
  const addTwoNumbers = (start = 1) => {
    let n = start;

    const api = {
      // Implement each function in your API
      add(inc: number = 1) {
        n += inc;
        return api;
      },

      print() {
        console.log(n);
        return api;
      },
    };
    return api;
  };

  // Which allows the same style of API as we saw in jQuery:
  addTwoNumbers(1).add(3).add().print().add(1);

  // Here's a similar example which uses a class:
  class AddNumbers {
    private n: number;

    constructor(start = 0) {
      this.n = start;
    }

    public add(inc = 1) {
      this.n = this.n + inc;
      return this;
    }

    public print() {
      console.log(this.n);
      return this;
    }
  }

  // Here it is in action:
  new AddNumbers(2).add(3).add().print().add(1);
  ```
## Working With Classes
- **Classes 101**
  ```ts
  class Vendor {
    name: string;

    constructor(name: string) {
      this.name = name;
    }

    greet() {
      return "Hello, welcome to " + this.name;
    }
  }

  // An instance can be created via the new keyword, and you can call methods and access properties from the object.
  const shop = new Vendor("Ye Olde Shop");
  console.log(shop.greet());

  // You can subclass an object. Here's a food cart which has a variety as well as a name:
  class FoodTruck extends Vendor {
    cuisine: string;

    constructor(name: string, cuisine: string) {
      super(name);
      this.cuisine = cuisine;
    }

    greet() {
      return "Hi, welcome to food truck " + this.name + ". We serve " + this.cuisine + " food.";
    }
  }

  // TypeScript will provide errors when you only use one parameter creating one:
  const nameOnlyTruck = new FoodTruck("Salome's Adobo");

  // Correctly passing in two arguments will let you create a new instance of the FoodTruck:
  const truck = new FoodTruck("Dave's Doritos", "junk");
  console.log(truck.greet());
  ```
- **This** When calling a method of a class, you generally expect it to refer to the current instance of the class.
  ```ts
  class Safe {
    contents: string;

    constructor(contents: string) {
      this.contents = contents;
    }

    printContents() {
      console.log(this.contents);
    }
  }

  const safe = new Safe("Crown Jewels");
  safe.printContents();

  // For example, if you use a reference to the func in another
  // object, and then call it through that - the this variable
  // has moved to refer to the hosting object:

  const customObjectCapturingThis = { contents: "http://gph.is/VxeHsW", print: safe.printContents };
  customObjectCapturingThis.print(); // Prints "http://gph.is/VxeHsW" - not "Crown Jewels"

  // This is tricky, because when dealing with callback APIs -
  // it can be very tempting to pass the function reference
  // directly. This can be worked around by creating a new
  // function at the call site.

  const objectNotCapturingThis = { contents: "N/A", print: () => safe.printContents() };
  objectNotCapturingThis.print();

  // There are a few ways to work around this problem. One
  // route is to force the binding of this to be the object
  // you originally intended via bind.

  const customObjectCapturingThisAgain = { contents: "N/A", print: safe.printContents.bind(safe) };
  customObjectCapturingThisAgain.print();

  // To work around an unexpected this context, you can also
  // change how you create functions in your class. By
  // creating a property which uses an arrow function, the
  // binding of this is done at a different time. Which makes
  // it more predictable for those less experienced with the
  // JavaScript runtime.

  class SafelyBoundSafe {
    contents: string;

    constructor(contents: string) {
      this.contents = contents;
    }

    printContents = () => {
      console.log(this.contents);
    };
  }

  // Now passing the function to another object
  // to run does not accidentally change this.

  const saferSafe = new SafelyBoundSafe("Golden Skull");
  saferSafe.printContents();

  const customObjectTryingToChangeThis = {
    contents: "http://gph.is/XLof62",
    print: saferSafe.printContents,
  };

  customObjectTryingToChangeThis.print();
  ```
- **Generic classes**
  ```ts
  // This example is mostly in TypeScript, because it is much
  // easier to understand this way first. At the end we'll
  // cover how to create the same class but using JSDoc instead.

  // Generic Classes are a way to say that a particular type
  // depends on another type. For example, here is a drawer
  // which can hold any sort of object, but only one type:

  class Drawer<ClothingType> {
    contents: ClothingType[] = [];

    add(object: ClothingType) {
      this.contents.push(object);
    }

    remove() {
      return this.contents.pop();
    }
  }

  // In order to use a Drawer, you will need another
  // type to work with:

  interface Sock {
    color: string;
  }

  interface TShirt {
    size: "s" | "m" | "l";
  }

  // We can create a Drawer just for socks by passing in the
  // type Sock when we create a new Drawer:
  const sockDrawer = new Drawer<Sock>();

  // Now we can add or remove socks to the drawer:
  sockDrawer.add({ color: "white" });
  const mySock = sockDrawer.remove();

  // As well as creating a drawer for TShirts:
  const tshirtDrawer = new Drawer<TShirt>();
  tshirtDrawer.add({ size: "m" });

  // If you're a bit eccentric, you could even create a drawer
  // which mixes Socks and TShirts by using a union:

  const mixedDrawer = new Drawer<Sock | TShirt>();

  // Creating a class like Drawer without the extra TypeScript
  // syntax requires using the template tag in JSDoc. In this
  // example we define the template variable, then provide
  // the properties on the class:

  // To have this work in the playground, you'll need to change
  // the settings to be a JavaScript file, and delete the
  // TypeScript code above

  /**
   * @template {{}} ClothingType
   */
  class Dresser {
    constructor() {
      /** @type {ClothingType[]} */
      this.contents = [];
    }

    /** @param {ClothingType} object */
    add(object) {
      this.contents.push(object);
    }

    /** @return {ClothingType} */
    remove() {
      return this.contents.pop();
    }
  }

  // Then we create a new type via JSDoc:

  /**
   * @typedef {Object} Coat An item of clothing
   * @property {string} color The colour for coat
   */

  // Then when we create a new instance of that class
  // we use @type to assign the variable as a Dresser
  // which handles Coats.

  /** @type {Dresser<Coat>} */
  const coatDresser = new Dresser();

  coatDresser.add({ color: "green" });
  const coat = coatDresser.remove();
  ```
- **Mixins**
  ```ts
  // Mixins are a faux-multiple inheritance pattern for classes
  // in JavaScript which TypeScript has support for. The pattern
  // allows you to create a class which is a merge of many
  // classes.

  // To get started, we need a type which we'll use to extend
  // other classes from. The main responsibility is to declare
  // that the type being passed in is a class.

  type Constructor = new (...args: any[]) => {};

  // Then we can create a series of classes which extend
  // the final class by wrapping it. This pattern works well
  // when similar objects have different capabilities.

  // This mixin adds a scale property, with getters and setters
  // for changing it with an encapsulated private property:

  function Scale<TBase extends Constructor>(Base: TBase) {
    return class extends Base {
      // Mixins may not declare private/protected properties
      // however, you can use ES2020 private fields
      _scale = 1;

      setScale(scale: number) {
        this._scale = scale;
      }

      get scale(): number {
        return this._scale;
      }
    };
  }

  // This mixin adds extra methods around alpha composition
  // something which modern computers use to create depth:

  function Alpha<TBase extends Constructor>(Base: TBase) {
    return class extends Base {
      alpha = 1;

      setHidden() {
        this.alpha = 0;
      }

      setVisible() {
        this.alpha = 1;
      }

      setAlpha(alpha: number) {
        this.alpha = alpha;
      }
    };
  }

  // A simple sprite base class which will then be extended:

  class Sprite {
    name = "";
    x = 0;
    y = 0;

    constructor(name: string) {
      this.name = name;
    }
  }

  // Here we create two different types of sprites
  // which have different capabilities:

  const ModernDisplaySprite = Alpha(Scale(Sprite));
  const EightBitSprite = Scale(Sprite);

  // Creating instances of these classes shows that
  // the objects have different sets of properties
  // and methods due to their mixins:

  const flappySprite = new ModernDisplaySprite("Bird");
  flappySprite.x = 10;
  flappySprite.y = 20;
  flappySprite.setVisible();
  flappySprite.setScale(0.8);
  console.log(flappySprite.scale);

  const gameBoySprite = new EightBitSprite("L block");
  gameBoySprite.setScale(0.3);

  // Fails because an EightBitSprite does not have
  // the mixin for changing alphas:
  gameBoySprite.setAlpha(0.5);

  // If you want to make more guarantees over the classes
  // which you wrap, you can use a constructor with generics.

  type GConstructor<T = {}> = new (...args: any[]) => T;

  // Now you can declare that this mixin can only be
  // applied when the base class is a certain shape.

  type Moveable = GConstructor<{ setXYAcceleration: (x: number, y: number) => void }>;

  // We can then create a mixin which relies on the function
  // present in the parameter to the GConstructor above.

  function Jumpable<TBase extends Moveable>(Base: TBase) {
    return class extends Base {
      jump() {
        // This mixin knows about setXYAcceleration now
        this.setXYAcceleration(0, 20);
      }
    };
  }

  // We cannot create this sprite until there is a class
  // in the mixin hierarchy which adds setXYAcceleration:
  const UserSprite = new Jumpable(ModernDisplaySprite);
  ```
## Modern JavaScript
***
## Recommended route
有，而且以你现在 **Python + Go + C++** 的背景，我不建议从“编程基础”开始学 TS。你真正需要补的是两层：

**JavaScript runtime 心智模型 → TypeScript 类型系统。**

官方 Handbook 自己也说，有其他语言经验的人通常可以很快掌握 JS 语法；Handbook 本身设计成几小时可以通读，而不是完整的 JS 教程。[TypeScript](https://www.typescriptlang.org/docs/handbook/intro.html?trk=public_post_comment-text&utm_source=chatgpt.com)

我会按下面路线走。

### 1. 先花 2–4 小时补 Modern JavaScript

不要一上来研究 TS 高级类型。你从 Go/C++ 过来，真正容易踩坑的是 **JS，而不是 TS**。

重点只学这些：

```
let / const
object / array
destructuring
spread ...
map / filter / reduce
closure
this
=== vs ==
null / undefined
truthy / falsy
Promise
async / await
ES modules (import/export)
event loop
```

尤其是：

**closure、Promise、event loop、object reference、`undefined`、module system。**

这些东西才会让一个 C++/Go programmer 感觉“这语言怎么这么怪”。

TypeScript 官方 Playground 本身就有 JavaScript Essentials、functions、classes、async/await、import/export 等可运行示例，很适合你快速扫，而不用看一门完整 JS 课程。[TypeScript](https://www.typescriptlang.org/play/?utm_source=chatgpt.com)

[TypeScript Playground](https://www.typescriptlang.org/play?utm_source=chatgpt.com)

---

### 2. 然后直接读官方 TypeScript Handbook

这是我最推荐你的主线资料：

[TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html?utm_source=chatgpt.com)

不要所有章节平均用力。

按这个顺序：

```
The Basics
    ↓
Everyday Types
    ↓
Narrowing          ★★★
    ↓
More on Functions
    ↓
Object Types
    ↓
Generics           ★★★
    ↓
keyof / typeof
    ↓
Indexed Access
    ↓
Conditional Types
    ↓
Mapped Types
    ↓
Modules
```

官方文档的 Handbook 和 Reference 本身也是这样分层的：前者面向日常使用，后者再深入 utility types、declaration merging、decorators 等。[TypeScript](https://www.typescriptlang.org/vo/docs/handbook/?utm_source=chatgpt.com)

对你而言，**Narrowing 特别值得认真学**。

比如：

```
type Result<T> =
    | { ok: true; value: T }
    | { ok: false; error: Error };

function handle(r: Result<string>) {
    if (r.ok) {
        console.log(r.value);
    } else {
        console.log(r.error);
    }
}
```

这时候你会开始理解 TS 为什么和 Go 的：

```
value, err := foo()
if err != nil {
    ...
}
```

思维方式很不一样。

---

### 3. 强烈推荐 Total TypeScript

如果官方 Handbook 是“教材”，这个更像 **LeetCode 式学 TS**：

[Total TypeScript 免费教程](https://www.totaltypescript.com/tutorials?utm_source=chatgpt.com)

尤其先做：

**Beginner's TypeScript**

它有 18 个 exercise，从 `any`、object parameter、optional property，一直到 union narrowing、utility types、async typing。[Total TypeScript](https://www.totaltypescript.com/tutorials/beginners-typescript?utm_source=chatgpt.com)

然后做：

```
Beginner's TypeScript
        ↓
Solving TypeScript Errors
        ↓
再根据需要：
React with TypeScript
Zod
Advanced TypeScript
```

它的 GitHub 版也可以直接：

```
npm install
npm run exercise
```

每题都有 `.problem` 和 `.solution`，测试通过就说明做对了。[GitHub](https://github.com/total-typescript/beginners-typescript-tutorial?utm_source=chatgpt.com)

你这种已经会三门语言的人，我觉得 **做题比看十几个小时视频有效得多**。

---

## 4. 不要太早学 React

这是我特别建议你避免的坑。

很多人的路径是：

```
不会 JS
  ↓
直接 React
  ↓
同时学习 JS + TS + JSX + React hooks + Next.js
  ↓
🤯
```

最后分不清：

```
哪些是 JavaScript
哪些是 TypeScript
哪些是 React
哪些是 Next.js
```

你反而可以利用自己的 Go 背景，**先用 TS 写 backend**。

例如做一个非常简单的：

```
HTTP API
WebSocket
CLI
JSON parser
并发调用几个 API
```

这样你只是在比较：

```
Go                 TypeScript

struct        ↔    interface/type
interface     ↔    structural typing
[]T           ↔    T[]
map[K]V       ↔    Map<K,V> / Record<K,V>
goroutine     ↔    Promise
channel       ↔    没有直接对应
error         ↔    throw/Result pattern
nil           ↔    null/undefined
generic       ↔    generic
```

学习速度会非常快。

---

## 5. 然后专门花时间理解 TS 最“非 Go/C++”的东西

这才是你学习 TS 真正有价值的部分。

比如：

```
type A = {
    name: string
}

type B = {
    name: string
    age: number
}

const b: B = {
    name: "Gavin",
    age: 30
}

const a: A = b;   // OK
```

为什么？

因为 TS 主要是 **structural typing**。

不是：

> B 有没有声明 implements A？

而是：

> B 的结构是否满足 A？

再进一步：

```
type Keys = keyof User;

type OptionalUser = Partial<User>;

type UserWithoutID = Omit<User, "id">;

type API<T> = {
    [K in keyof T]: Promise<T[K]>
};
```

你会逐渐发现 TS 类型系统不是：

> C++/Go 的类型系统换了个语法。

而更接近一个**对 JavaScript object shape 做编译期计算的小语言**。

官方 Playground 甚至专门把 Conditional Types、Discriminated Types、Indexed Types、Mapped Types、Structural Typing、Narrowing 等作为独立示例。[TypeScript](https://www.typescriptlang.org/zh/play/?utm_source=chatgpt.com)

这部分才值得你投入时间。

---

## 6. 最后做一个真正的项目

以你的技术栈，我会建议一个非常具体的练习：

```
              Browser
                 │
        React + TypeScript
                 │
                 ↓
        TypeScript API/BFF
                 │
          ┌──────┴──────┐
          ↓             ↓
       Go API        Python API
                         │
                      ML / AI
```

例如做一个小型 AI application。

你已有的：

**Python → AI/data**

**Go → backend service**

然后新写：

**TS → Web UI + 一个 Node API/BFF**

你马上就会碰到 TS 真正解决的问题：

```
API response typing
union types
Promise
runtime validation
JSON ≠ TypeScript type
shared schema
frontend state
React props
error handling
```

这比单独刷 TS syntax 有效得多。

---

### 我给你的最短学习路线

如果每天 1～2 小时：

**Day 1–2**  
Modern JS + Playground

**Day 3–5**  
TypeScript Handbook：Everyday Types / Narrowing / Functions / Objects / Generics

**Day 6–7**  
Total TypeScript exercises

**Week 2**  
直接做一个 TS API，然后 React + TS 前端。

到这里你应该已经能正常写生产级 TS 了。后面的 conditional/mapped/template literal types 等，**遇到再深入，不需要先学完**。

还有一个很好的付费交互课程是 Execute Program，它目前把内容拆成 Modern JavaScript、JavaScript Concurrency、TypeScript Basics、Everyday TypeScript、Advanced TypeScript；如果你喜欢“解释一点 → 马上写代码 → 间隔复习”的方式，它做得很好，但目前是 $39/月，所以我会先把免费的官方 Handbook + Total TypeScript 用完。[执行程序](https://www.executeprogram.com/?utm_source=chatgpt.com)

以你的背景，我甚至建议**不要买“零基础 TypeScript 课程”**。你已经不需要别人花两小时给你解释 variable、loop、class 了。真正值得花时间的是 **JS runtime + TS type system + Web ecosystem**。

<!-- related -->
# Related
- [Flutter](03-09_flutter.md)
- [cgo](../2020/02-11_cgo.md)
***
<!-- /related -->
