import { Tilemap } from '@pixi/tilemap';
import { Loader } from '@pixi/loaders';

const textureRoot = '/texture/';

// Read in simulation state from JSON
const { agents, tile_width, tile_set, map } = await fetch('./json/simulation.json')
    .then(response => response.json())
    .catch(error => console.error('Error loading JSON: ', error));

// Initialize application
const app = new PIXI.Application();
await app.init({ width: tile_width * map.length, height: tile_width * map[0].length });
document.body.appendChild(app.canvas);

// Initialize agents
agents.forEach(async (agent) => {
    const texture = textureRoot + agent.texture;

    await PIXI.Assets.load(texture);
    agent.sprite = PIXI.Sprite.from(texture);

    agent.sprite.x = agent.position.x + tile_width / 2;
    agent.sprite.y = agent.position.y + tile_width / 2;

    app.stage.addChild(agent.sprite);
});

// Initialize tile set
Loader.shared.load(function onTilesetLoaded() {
    const tilemap = new Tilemap([Texture.from(textureRoot + tile_set[0])]);

    for (let i = 0; i < map.length; i++) {
        for (let j = 0; j < map[0].length; j++) {
            tilemap.tile(textureRoot + tile_set[map[i][j]], i, j);
        }
    }
})

// Animate Rendering
let elapsed = 0.0;

app.ticker.add((ticker) => {
    // elapsed += ticker.deltaTime;

    // simulation.agents.forEach(async agent => {
    //     agent.sprite.x = agent.position.x + Math.cos(elapsed / 50.0) * 100;
    // })

});