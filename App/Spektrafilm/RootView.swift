import SpektraFilm
import SwiftUI

/// Placeholder shell.
///
/// Deliberately thin: it exists so the app target links the engine and so CI proves the bundled
/// profiles and spectral tables load from a real app bundle, which a `swift test` run on macOS does
/// not exercise. The editor UI replaces this.
struct RootView: View {
    @State private var stocks: [Profile] = []
    @State private var loadFailure: String?

    var body: some View {
        NavigationStack {
            List {
                if let loadFailure {
                    Section("Engine") {
                        Label(loadFailure, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                    }
                } else {
                    Section("Film stocks") {
                        ForEach(stocks, id: \.info.stock) { profile in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(profile.info.displayName)
                                Text(profile.isPositive ? "Positive" : "Negative")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Spektrafilm")
        }
        .task {
            do {
                stocks = try ProfileLibrary.filmStocks
            } catch {
                loadFailure = String(describing: error)
            }
        }
    }
}

#Preview {
    RootView()
}
