import { describe, expect, it } from 'vitest';
import { isLanguageAvailable } from '../../../src/core/tree-sitter/parser-loader.js';
import { SupportedLanguages } from '../../../src/config/supported-languages.js';
import { getRelationships, useFixture } from './helpers.js';

const swiftAvailable = isLanguageAvailable(SupportedLanguages.Swift);

// Each view file names its model type only inside the property wrapper, so any
// USES edge to Svc / Product can only come from the wrapper.
const view = (name: string, property: string, imports = 'import SwiftUI') => `${imports}

struct ${name}: View {
    ${property}
    var body: some View { Text("x") }
}
`;

const views: Array<[file: string, property: string, target: string | null]> = [
  ['EnvView.swift', '@Environment(Svc.self) private var svc', 'Svc'],
  ['QualifiedEnvView.swift', '@Environment(Models.Svc.self) private var svc', 'Svc'],
  ['ModuleWrapperView.swift', '@SwiftUI.Environment(Svc.self) private var svc', 'Svc'],
  ['QueryView.swift', '@Query(sort: \\Product.name) private var products: [Product]', 'Product'],
  ['BareQueryView.swift', '@Query var products: [Product]', 'Product'],
  ['OptionalQueryView.swift', '@Query var products: [Product]?', 'Product'],
  ['DefaultQueryView.swift', '@Query var products: [Product] = []', 'Product'],
  ['GenericQueryView.swift', '@Query var products: Array<Product>', 'Product'],
  ['QualifiedQueryView.swift', '@Query var products: [Models.Product]', 'Product'],
  ['KeyPathView.swift', '@Environment(\\.colorScheme) private var scheme', null],
  ['SelfKeyPathView.swift', '@Environment(\\.self) private var env', null],
];

describe.skipIf(!swiftAvailable)('Swift @Environment / @Query type dependencies', () => {
  const result = useFixture('swift-property-wrapper-deps', {
    'Svc.swift': 'class Svc {\n    func run() {}\n}\n',
    'Product.swift': 'class Product {\n    var name = ""\n}\n',
    ...Object.fromEntries(
      views.map(([file, property]) => [
        file,
        view(file.replace('.swift', ''), property, 'import SwiftUI\nimport SwiftData'),
      ]),
    ),
  });

  const modelUsesFrom = (file: string) =>
    getRelationships(result(), 'USES')
      .filter((e) => e.sourceFilePath === file && ['Svc', 'Product'].includes(e.target))
      .map((e) => ({ source: e.source, target: e.target, targetFilePath: e.targetFilePath }));

  it.each(views)('%s with `%s` depends on %s', (file, _property, target) => {
    const expected =
      target === null
        ? []
        : [{ source: file.replace('.swift', ''), target, targetFilePath: `${target}.swift` }];
    expect(modelUsesFrom(file)).toEqual(expected);
  });
});
