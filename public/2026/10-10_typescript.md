# ___Typescript___
***

# Node and TypeScript Toolchain
## Links
  - [Node.js](https://nodejs.org/)
  - [TypeScript](https://www.typescriptlang.org/docs/)
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
## Full teardown
  ```sh
  rm -rf ~/.fnm ~/.local/share/fnm
  rm -rf ~/.local/share/pnpm ~/.cache/pnpm
  rm -rf ~/.npm ~/.npmrc
  # then strip the fnm / PNPM_HOME blocks from ~/.bashrc
  ```
***

# Hello World
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
# Modern JavaScript
- **Objects and arrays**
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
- **Generic functions**
  ```ts
  // Generics provide a way to use Types as variables in other types. Meta.
  function wrapInArray<Type>(input: Type): Type[] {
    return [input];
  }

  const stringArray = wrapInArray("hello generics");
  const numberArray = wrapInArray(123);
  // You can also skip the generic inference by adding the type yourself also:
  const stringArray2 = wrapInArray<string>("");

  // wrapInArray allows any type to be used, however there
  // are cases when you need to only allow a subset of types.
  // In these cases you can say the type has to extend a
  // particular type.

  interface Drawable {
    draw: () => void;
  }

  // This function takes a set of objects which have a function
  // for drawing to the screen
  function renderToScreen<Type extends Drawable>(input: Type[]) {
    input.forEach((i) => i.draw());
  }

  const objectsWithDraw = [{ draw: () => {} }, { draw: () => {} }];
  renderToScreen(objectsWithDraw);

  // It will fail if draw is missing:

  renderToScreen([{}, { draw: () => {} }]);

  // Generics can start to look complicated when you have
  // multiple variables. Here is an example of a caching
  // function that lets you have different sets of input types
  // and caches.

  interface CacheHost {
    save: (a: any) => void;
  }

  function addObjectToCache<Type, Cache extends CacheHost>(obj: Type, cache: Cache): Cache {
    cache.save(obj);
    return cache;
  }

  // This is the same as above, but with an extra parameter.
  // Note: to make this work though, we had to use an any. This
  // can be worked out by using a generic interface.

  interface CacheHostGeneric<ContentType> {
    save: (a: ContentType) => void;
  }

  // Now when the CacheHostGeneric is used, you need to tell
  // it what ContentType is.

  function addTypedObjectToCache<Type, Cache extends CacheHostGeneric<Type>>(obj: Type, cache: Cache): Cache {
    cache.save(obj);
    return cache;
  }
  ```
不要一上来研究 TS 高级类型。你从 Go/C++ 过来，真正容易踩坑的是 JS，而不是 TS。
重点只学这些：
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

尤其是：
closure、Promise、event loop、object reference、undefined、module system。
# TypeScript Handbook
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
