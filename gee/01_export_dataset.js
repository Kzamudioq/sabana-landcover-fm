/**
 * sabana-landcover-fm · 01_export_dataset.js
 * Google Earth Engine Code Editor (https://code.earthengine.google.com)
 *
 * Exports everything the experiment needs for the Sabana de Bogotá, year 2021
 * (the year of the ESA WorldCover v200 labels), all on the same 10 m grid:
 *
 *   s2_2021_hls6.tif     6 bands int16  Sentinel-2 L2A median, Cloud Score+ masked
 *                                       B2 B3 B4 B8A B11 B12 (the Prithvi/HLS band set), reflectance x 10000
 *   labels_2021.tif      3 bands uint8  b1 WorldCover 2021 -> 7-class scheme (training labels)
 *                                       b2 Dynamic World 2021 -> 7-class scheme (baseline map)
 *                                       b3 split: 1 train, 2 val, 3 test (5 km spatial blocks)
 *                                       255 = no data in every band
 *   split_grid.geojson   the 5 km blocks with their split, for the record
 *   ref_points.csv       id, lon, lat ONLY -> label these blind (docs/protocolo_verificacion.md)
 *   ref_strata.csv       id, stratum (WorldCover class) -> keep closed until labelling is done
 *   strata_pixels.csv    stratum, pixels: size of each stratum inside the test blocks
 *
 * The 7 classes: 0 water, 1 trees, 2 shrubland, 3 grass/herbaceous, 4 cropland, 5 built-up, 6 bare.
 * Before running: set the Cloud project (avatar > Change Cloud Project). Then Run, and
 * press RUN on each export in the Tasks tab. Everything lands in Drive/sabana_lc.
 */

// ------------------------------------------------------------------ settings
var YEAR = 2021;
var CRS = 'EPSG:32618';           // UTM 18N: metric grid for chips; final areas are ellipsoidal anyway
var SCALE = 10;
var FOLDER = 'sabana_lc';
var BLOCK_M = 5000;               // spatial blocks for train/val/test (no leakage between neighbours)
var SEED = 42;
var POINTS_PER_STRATUM = 70;      // 7 strata -> 490 reference points in the test blocks
var CS_THRESHOLD = 0.60;          // Cloud Score+ "cs_cdf": keep pixels clearer than this

// Sabana de Bogotá: Bogotá urban area, Cerros Orientales and the neighbouring municipalities
// (Chía, Cota, Funza, Mosquera, Madrid, Soacha, La Calera). Replace with your own polygon if you like.
var AOI = ee.Geometry.Rectangle([-74.32, 4.47, -73.94, 4.92], null, false);
Map.centerObject(AOI, 10);

// ------------------------------------------------------------------ Sentinel-2 composite
var BANDS = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'];
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(AOI)
    .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01');
var cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var composite = s2.linkCollection(cs, ['cs_cdf'])
    .map(function (img) { return img.updateMask(img.select('cs_cdf').gte(CS_THRESHOLD)); })
    .select(BANDS)
    .median()
    .round().toInt16();
var clearObs = s2.linkCollection(cs, ['cs_cdf'])
    .map(function (img) { return img.select('cs_cdf').gte(CS_THRESHOLD).selfMask(); })
    .count();
print('Sentinel-2 scenes ' + YEAR, s2.size());
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 200, max: 1800}, 'S2 ' + YEAR + ' RGB');
Map.addLayer(clearObs.clip(AOI), {min: 0, max: 40, palette: ['b2182b', 'f7f7f7', '2166ac']}, 'clear observations', false);

// ------------------------------------------------------------------ labels (7-class scheme)
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
//          WorldCover: 10 trees,20 shrub,30 grass,40 crop,50 built,60 bare,70 snow,80 water,90 wetland,95 mangrove,100 moss
var wc7 = wc.remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                   [1, 2, 3, 4, 5, 6, 255, 0, 3, 1, 3], 255).toUint8();

var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(AOI)
    .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01');
var DW = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'];
var dwTop = dwCol.select(DW).mean().toArray().arrayArgmax().arrayGet([0]);
//            DW: 0 water,1 trees,2 grass,3 flooded veg,4 crops,5 shrub,6 built,7 bare,8 snow
var dw7 = dwTop.remap([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 3, 3, 4, 2, 5, 6, 255], 255).toUint8();

var palette7 = ['3a7bd5', '1f6e43', '9bbf4a', 'd8e27a', 'f2b134', 'd7263d', 'b8a99a'];
Map.addLayer(wc7.clip(AOI), {min: 0, max: 6, palette: palette7}, 'WorldCover 2021 (7 classes)');
Map.addLayer(dw7.clip(AOI), {min: 0, max: 6, palette: palette7}, 'Dynamic World 2021 (7 classes)', false);

// ------------------------------------------------------------------ spatial split
var proj = ee.Projection(CRS).atScale(BLOCK_M);
var grid = AOI.coveringGrid(proj)
    .randomColumn('r', SEED)
    .map(function (f) {
      var r = ee.Number(f.get('r'));
      var split = ee.Algorithms.If(r.lt(0.2), 3, ee.Algorithms.If(r.lt(0.3), 2, 1));
      return f.set('split', split);
    });
var splitImg = grid.reduceToImage(['split'], ee.Reducer.first()).toUint8();
Map.addLayer(grid.style({color: '000000', fillColor: '00000000', width: 1}), {}, 'blocks', false);
Map.addLayer(splitImg.clip(AOI), {min: 1, max: 3, palette: ['cccccc', 'f2b134', 'd7263d']}, 'split (grey train, yellow val, red test)', false);

var valid = composite.select('B4').mask().and(wc7.neq(255));
var labels = ee.Image.cat(wc7.rename('worldcover'), dw7.rename('dynamicworld'), splitImg.rename('split'))
    .updateMask(valid).unmask(255).toUint8();

// ------------------------------------------------------------------ reference sample (test blocks only)
var testMask = splitImg.eq(3).and(wc7.neq(255)).and(composite.select('B4').mask());
var strata = wc7.updateMask(testMask).rename('stratum');
var sample = strata.stratifiedSample({
  numPoints: POINTS_PER_STRATUM, classBand: 'stratum', region: AOI, scale: SCALE,
  projection: CRS, seed: SEED, geometries: true, dropNulls: true, tileScale: 4
});
sample = ee.FeatureCollection(sample.randomColumn('shuffle', SEED + 1).sort('shuffle')
    .toList(1000).zip(ee.List.sequence(1, 1000))
    .map(function (pair) {
      pair = ee.List(pair);
      var f = ee.Feature(pair.get(0));
      var c = f.geometry().transform('EPSG:4326', 0.01).coordinates();
      return f.set({id: pair.get(1), lon: c.get(0), lat: c.get(1)});
    }));
print('Reference points per stratum', sample.aggregate_histogram('stratum'));

var stratumPixels = strata.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(), geometry: AOI, scale: SCALE, crs: CRS,
  maxPixels: 1e10, tileScale: 4
}).get('stratum');
var stratumTable = ee.FeatureCollection(ee.Dictionary(stratumPixels).keys().map(function (k) {
  return ee.Feature(null, {stratum: ee.Number.parse(k), pixels: ee.Dictionary(stratumPixels).get(k)});
}));
print('Pixels per stratum in the test blocks', stratumTable);

// ------------------------------------------------------------------ exports
function exportImage(img, name) {
  Export.image.toDrive({
    image: img, description: name, folder: FOLDER, fileNamePrefix: name, region: AOI,
    crs: CRS, scale: SCALE, maxPixels: 1e10, fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
}
exportImage(composite.unmask(-32768).clip(AOI), 's2_2021_hls6');
exportImage(labels.clip(AOI), 'labels_2021');

Export.table.toDrive({collection: grid, description: 'split_grid', folder: FOLDER, fileFormat: 'GeoJSON'});
Export.table.toDrive({collection: sample, description: 'ref_points', folder: FOLDER,
                      fileFormat: 'CSV', selectors: ['id', 'lon', 'lat']});
Export.table.toDrive({collection: sample, description: 'ref_strata', folder: FOLDER,
                      fileFormat: 'CSV', selectors: ['id', 'stratum']});
Export.table.toDrive({collection: stratumTable, description: 'strata_pixels', folder: FOLDER,
                      fileFormat: 'CSV', selectors: ['stratum', 'pixels']});
