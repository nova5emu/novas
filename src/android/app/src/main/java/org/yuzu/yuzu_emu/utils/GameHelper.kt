// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

package org.yuzu.yuzu_emu.utils

import android.content.SharedPreferences
import android.net.Uri
import android.os.Environment
import android.provider.DocumentsContract
import androidx.preference.PreferenceManager
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import org.yuzu.yuzu_emu.NativeLibrary
import org.yuzu.yuzu_emu.YuzuApplication
import org.yuzu.yuzu_emu.model.Game
import org.yuzu.yuzu_emu.model.GameDir
import org.yuzu.yuzu_emu.model.MinimalDocumentFile
import androidx.core.content.edit
import androidx.core.net.toUri

object GameHelper {
    private const val KEY_OLD_GAME_PATH = "game_path"
    const val KEY_GAMES = "Games"

    var cachedGameList = mutableListOf<Game>()

    private lateinit var preferences: SharedPreferences

    fun getGames(): List<Game> {
        val games = mutableListOf<Game>()
        val gamesByProgramId = mutableMapOf<String, Game>()
        val context = YuzuApplication.appContext
        preferences = PreferenceManager.getDefaultSharedPreferences(context)

        val gameDirs = mutableListOf<GameDir>()
        val oldGamesDir = preferences.getString(KEY_OLD_GAME_PATH, "") ?: ""
        if (oldGamesDir.isNotEmpty()) {
            gameDirs.add(GameDir(oldGamesDir, true))
            preferences.edit() { remove(KEY_OLD_GAME_PATH) }
        }
        gameDirs.addAll(NativeConfig.getGameDirs())

        // Ensure keys are loaded so that ROM metadata can be decrypted.
        NativeLibrary.reloadKeys()

        // Reset metadata so we don't use stale information
        GameMetadata.resetMetadata()

        // Remove previous filesystem provider information so we can get up to date version info
        NativeLibrary.clearFilesystemProvider()

        val mountedContainerUris = mutableSetOf<String>()
        looseGamesNeedAllFilesAccess = false
        mountExternalContentDirectories(mountedContainerUris)

        val badDirs = mutableListOf<Int>()
        gameDirs.forEachIndexed { index: Int, gameDir: GameDir ->
            val gameDirUri = gameDir.uriString.toUri()
            val isValid = FileUtil.isTreeUriValid(gameDirUri)
            if (isValid) {
                // NovaS: deep enough for a loose dump's base/exefs under a game folder (as on PC).
                val scanDepth = if (gameDir.deepScan) 5 else 1

                addGamesRecursive(
                    games,
                    gamesByProgramId,
                    FileUtil.listFiles(gameDirUri),
                    scanDepth,
                    mountedContainerUris
                )
            } else {
                badDirs.add(index)
            }
        }

        // Remove all game dirs with insufficient permissions from config
        if (badDirs.isNotEmpty()) {
            var offset = 0
            badDirs.forEach {
                gameDirs.removeAt(it - offset)
                offset++
            }
        }
        NativeConfig.setGameDirs(gameDirs.toTypedArray())

        // Cache list of games found on disk
        val serializedGames = mutableSetOf<String>()
        games.forEach {
            serializedGames.add(Json.encodeToString(it))
        }
        preferences.edit() {
            remove(KEY_GAMES)
                .putStringSet(KEY_GAMES, serializedGames)
        }

        cachedGameList = games.toMutableList()
        return games.toList()
    }

    fun restoreContentForGame(game: Game) {
        NativeLibrary.reloadKeys()

        val mountedContainerUris = mutableSetOf<String>()
        mountExternalContentDirectories(mountedContainerUris)
        mountGameFolderContent(Uri.parse(game.path), mountedContainerUris)
        NativeLibrary.addFileToFilesystemProvider(game.path)
    }

    // File extensions considered as external content, buuut should
    // be done better imo.
    private val externalContentExtensions = setOf("nsp", "xci", "nsz", "xcz")

    private fun scanContentContainersRecursive(
        files: Array<MinimalDocumentFile>,
        depth: Int,
        onContainerFound: (MinimalDocumentFile) -> Unit
    ) {
        if (depth <= 0) {
            return
        }

        files.forEach {
            if (it.isDirectory) {
                scanContentContainersRecursive(
                    FileUtil.listFiles(it.uri),
                    depth - 1,
                    onContainerFound
                )
            } else {
                val extension = FileUtil.getExtension(it.uri).lowercase()
                if (externalContentExtensions.contains(extension)) {
                    onContainerFound(it)
                }
            }
        }
    }

    private fun addGamesRecursive(
        games: MutableList<Game>,
        gamesByProgramId: MutableMap<String, Game>,
        files: Array<MinimalDocumentFile>,
        depth: Int,
        mountedContainerUris: MutableSet<String>
    ) {
        if (depth <= 0) {
            return
        }

        files.forEach {
            if (it.isDirectory) {
                val dirName = it.filename.lowercase()
                // NovaS: an extracted game (exefs with main and main.npdm), or a loose dump's base
                // (base/exefs, next to updates/ and dlc/) as on PC. Its update's and DLC's exefs are
                // not games of their own, and romfs holds only the game's data.
                if (dirName in LOOSE_DUMP_SKIPPED) {
                    return@forEach
                }
                val children = FileUtil.listFiles(it.uri)
                if (dirName == "exefs") {
                    addLooseGame(games, gamesByProgramId, children, null)
                    return@forEach
                }
                children.firstOrNull { child -> child.isDirectory && child.filename.equals("base", true) }
                    ?.let { base ->
                        FileUtil.listFiles(base.uri)
                            .firstOrNull { child -> child.isDirectory && child.filename.equals("exefs", true) }
                            ?.let { exefs ->
                                addLooseGame(games, gamesByProgramId, FileUtil.listFiles(exefs.uri), it.filename)
                            }
                        return@forEach
                    }
                addGamesRecursive(
                    games,
                    gamesByProgramId,
                    children,
                    depth - 1,
                    mountedContainerUris
                )
            } else {
                val extension = FileUtil.getExtension(it.uri).lowercase()
                val filePath = it.uri.toString()

                val mountedContainer = externalContentExtensions.contains(extension) &&
                    mountedContainerUris.add(filePath)
                if (mountedContainer) {
                    NativeLibrary.addGameFolderFileToFilesystemProvider(filePath)
                }

                if (Game.extensions.contains(extension)) {
                    val game = getGame(it.uri, true, false)
                    if (game != null) {
                        games.add(game)
                        if (game.programId != "0") {
                            gamesByProgramId[game.programId] = game
                        }
                    } else if (mountedContainer) {
                        GameMetadata.getProgramId(filePath).toLongOrNull()?.let { programId ->
                            gamesByProgramId[(programId and 0x800L.inv()).toString()]
                        }?.let { existingGame ->
                            NativeLibrary.getPatchesForFile(existingGame.path, existingGame.programId)
                            existingGame.version = GameMetadata.getVersion(
                                existingGame.path,
                                true
                            )
                            GameIconUtils.refreshGameIcon(existingGame)
                        }
                    }
                }
            }
        }
    }

    // NovaS: folders a game scan does not enter: a loose dump's update and DLC contents and every
    // game's data (romfs holds thousands of files).
    private val LOOSE_DUMP_SKIPPED = setOf("updates", "update", "dlc", "romfs")

    /**
     * True when the last scan found an extracted game it could not open because NovaS lacks
     * access to all files (an extracted game is a folder the engine walks by path).
     */
    var looseGamesNeedAllFilesAccess = false
        private set

    // NovaS: an extracted game's ExeFS (main and main.npdm), booted by its main as on PC. The engine
    // walks the folders around it (romfs next to exefs, updates/ next to base/), which a content
    // URI cannot do, so it gets the file's real path.
    private fun addLooseGame(
        games: MutableList<Game>,
        gamesByProgramId: MutableMap<String, Game>,
        exefsFiles: Array<MinimalDocumentFile>,
        folderName: String?
    ) {
        val main = exefsFiles.firstOrNull { !it.isDirectory && it.filename == "main" } ?: return
        if (exefsFiles.none { !it.isDirectory && it.filename == "main.npdm" }) {
            return
        }
        val path = realPathOf(main.uri) ?: return
        val file = File(path)
        if (!file.canRead()) {
            looseGamesNeedAllFilesAccess = true
            Log.warning("[GameHelper] extracted game at $path needs access to all files")
            return
        }
        val exefsDir = file.parentFile
        val gameDir = exefsDir?.parentFile
        val name = folderName ?: if (gameDir?.name.equals("base", true)) {
            gameDir?.parentFile?.name
        } else {
            gameDir?.name
        }
        val game = getGame(Uri.fromFile(file), true, false, name?.let { NovaSCovers.tidyName(it) })
            ?: return
        if (games.none { it.path == game.path }) {
            games.add(game)
            if (game.programId != "0") {
                gamesByProgramId[game.programId] = game
            }
        }
    }

    // NovaS: the real path of a document from the phone's own storage or an SD card
    // ("primary:Games/x" -> /storage/emulated/0/Games/x), or null for other providers.
    private fun realPathOf(uri: Uri): String? {
        if (uri.scheme == "file") {
            return uri.path
        }
        val id = try {
            DocumentsContract.getDocumentId(uri)
        } catch (_: Exception) {
            return null
        }
        if (uri.authority != "com.android.externalstorage.documents") {
            return null
        }
        val colon = id.indexOf(':')
        if (colon < 0) {
            return null
        }
        val volume = id.substring(0, colon)
        val relative = id.substring(colon + 1)
        val root = if (volume.equals("primary", true)) {
            Environment.getExternalStorageDirectory().path
        } else {
            "/storage/$volume"
        }
        return if (relative.isEmpty()) root else "$root/$relative"
    }

    private fun mountExternalContentDirectories(mountedContainerUris: MutableSet<String>) {
        val uniqueExternalContentDirs = linkedSetOf<String>()
        NativeConfig.getExternalContentDirs().forEach { externalDir ->
            if (externalDir.isNotEmpty()) {
                uniqueExternalContentDirs.add(externalDir)
            }
        }

        for (externalDir in uniqueExternalContentDirs) {
            val externalDirUri = externalDir.toUri()
            if (FileUtil.isTreeUriValid(externalDirUri)) {
                scanContentContainersRecursive(FileUtil.listFiles(externalDirUri), 3) {
                    val containerUri = it.uri.toString()
                    if (mountedContainerUris.add(containerUri)) {
                        NativeLibrary.addFileToFilesystemProvider(containerUri)
                    }
                }
            }
        }
    }

    private fun mountGameFolderContent(gameUri: Uri, mountedContainerUris: MutableSet<String>) {
        if (gameUri.scheme == "content") {
            val parentUri = getParentDocumentUri(gameUri) ?: return
            scanContentContainersRecursive(FileUtil.listFiles(parentUri), 1) {
                val containerUri = it.uri.toString()
                if (mountedContainerUris.add(containerUri)) {
                    NativeLibrary.addGameFolderFileToFilesystemProvider(containerUri)
                }
            }
            return
        }

        val gameFile = File(gameUri.path ?: gameUri.toString())
        val parentDir = gameFile.parentFile ?: return
        parentDir.listFiles()?.forEach { sibling ->
            if (!sibling.isFile) {
                return@forEach
            }

            val extension = sibling.extension.lowercase()
            if (externalContentExtensions.contains(extension)) {
                val containerUri = Uri.fromFile(sibling).toString()
                if (mountedContainerUris.add(containerUri)) {
                    NativeLibrary.addGameFolderFileToFilesystemProvider(containerUri)
                }
            }
        }
    }

    private fun getParentDocumentUri(uri: Uri): Uri? {
        return try {
            val documentId = DocumentsContract.getDocumentId(uri)
            val separatorIndex = documentId.lastIndexOf('/')
            if (separatorIndex == -1) {
                null
            } else {
                val parentDocumentId = documentId.substring(0, separatorIndex)
                DocumentsContract.buildDocumentUriUsingTree(uri, parentDocumentId)
            }
        } catch (_: Exception) {
            null
        }
    }

    fun getGame(
        uri: Uri,
        addedToLibrary: Boolean,
        registerFilesystemProvider: Boolean = true,
        fallbackName: String? = null
    ): Game? {
        // NovaS: an extracted game goes to the engine as a plain path (it has no content URI form).
        val filePath = if (uri.scheme == "file") (uri.path ?: return null) else uri.toString()
        if (!GameMetadata.getIsValid(filePath)) {
            return null
        }

        if (registerFilesystemProvider) {
            // Needed to update installed content information
            NativeLibrary.addFileToFilesystemProvider(filePath)
        }

        var name = GameMetadata.getTitle(filePath)

        // If the game's title field is empty, use the filename (a loose dump: its folder's name).
        if (name.isEmpty()) {
            name = fallbackName ?: FileUtil.getFilename(uri)
        }
        var programId = GameMetadata.getProgramId(filePath)

        // If the game's ID field is empty, use the filename without extension.
        if (programId.isEmpty()) {
            programId = name.substringBeforeLast(".")
        }

        // NovaS: the known Switch 2 game's name, else the file name without its tags (as on PC).
        name = NovaSCovers.tidyName(name)

        val newGame = Game(
            name,
            filePath,
            programId,
            GameMetadata.getDeveloper(filePath),
            GameMetadata.getVersion(filePath, false),
            GameMetadata.getIsHomebrew(filePath)
        )

        if (addedToLibrary) {
            val addedTime = preferences.getLong(newGame.keyAddedToLibraryTime, 0L)
            if (addedTime == 0L) {
                preferences.edit()
                    .putLong(newGame.keyAddedToLibraryTime, System.currentTimeMillis())
                    .apply()
            }
        }

        return newGame
    }
}
