// SPDX-License-Identifier: GPL-3.0-or-later

package org.yuzu.yuzu_emu.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.text.Normalizer
import org.yuzu.yuzu_emu.YuzuApplication

// NovaS: covers and tidy names for known Switch 2 games, the same catalog as the PC launcher (SwitchCovers.cs),
// shipped in assets/novas-covers. A game matches when its name contains one of the entry's names, compared without
// accents, spaces or punctuation, so "[www.site.com] - pokopia_v2-0-1" is Pokémon Pokopia.
object NovaSCovers {
    private class Entry(val id: String, val name: String, vararg val aliases: String)

    // Most specific names first: the first match wins.
    private val catalog = listOf(
        Entry("zelda-ocarina-of-time", "The Legend of Zelda: Ocarina of Time", "Ocarina of Time"),
        Entry("hyrule-warriors-age-of-imprisonment", "Hyrule Warriors: Age of Imprisonment", "Age of Imprisonment", "Era del destierro", "Hyrule Warriors"),
        Entry("mario-kart-world", "Mario Kart World", "Mario Kart World"),
        Entry("donkey-kong-bananza", "Donkey Kong Bananza", "Donkey Kong Bananza", "DK Bananza"),
        Entry("pokemon-legends-z-a", "Pokémon Legends: Z-A", "Legends Z-A", "Leyendas Z-A", "Legends ZA", "Leyendas ZA"),
        Entry("pokemon-pokopia", "Pokémon Pokopia", "Pokopia"),
        Entry("nintendo-switch-sports-resort", "Nintendo Switch Sports Resort", "Switch Sports Resort", "Sports Resort"),
        Entry("splatoon-raiders", "Splatoon Raiders", "Splatoon Raiders"),
        Entry("kirby-air-riders", "Kirby Air Riders", "Kirby Air Riders", "Air Riders"),
        Entry("super-mario-bros-wonder", "Super Mario Bros. Wonder", "Mario Bros Wonder", "Mario Wonder"),
        Entry("mario-tennis-fever", "Mario Tennis Fever", "Mario Tennis Fever"),
        Entry("super-mario-galaxy-2", "Super Mario Galaxy 2", "Mario Galaxy 2", "Mario Galaxy"),
        Entry("fire-emblem-fortunes-weave", "Fire Emblem: Fortune's Weave", "Fortune's Weave", "Fortunes Weave"),
        Entry("yoshi-and-the-mysterious-book", "Yoshi and the Mysterious Book", "Mysterious Book", "Libro misterioso", "Yoshi"),
        Entry("pragmata", "Pragmata", "Pragmata"),
        Entry("cyberpunk-2077", "Cyberpunk 2077: Ultimate Edition", "Cyberpunk"),
        Entry("deltarune", "Deltarune", "Deltarune"),
        Entry("fast-fusion", "Fast Fusion", "Fast Fusion"),
        Entry("red-dead-redemption", "Red Dead Redemption", "Red Dead Redemption"),
        Entry("resident-evil-4", "Resident Evil 4", "Resident Evil 4", "Biohazard 4"),
        Entry("metroid-prime-4-beyond", "Metroid Prime 4: Beyond", "Metroid Prime 4"),
        Entry("star-fox", "Star Fox", "Star Fox", "Starfox"),
        Entry("drag-x-drive", "Drag x Drive", "Drag x Drive", "Drag Drive"),
    )

    // "Pokémon Legends: Z-A" -> "pokemonlegendsza".
    private fun key(text: String): String =
        Normalizer.normalize(text, Normalizer.Form.NFD)
            .filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() && it.isLetterOrDigit() }
            .lowercase()

    private fun find(name: String): Entry? {
        val k = key(name)
        return catalog.firstOrNull { entry -> entry.aliases.any { k.contains(key(it)) } }
    }

    // The known game's tidy name, or the name without the "[site]" / "(region)" tags and the file extension.
    fun tidyName(name: String): String {
        find(name)?.let { return it.name }
        var clean = name.substringBeforeLast('.', name)
        clean = clean.replace(Regex("""\[[^\]]*]|\([^)]*\)"""), " ").replace('_', ' ')
        clean = clean.trim(' ', '-').replace(Regex("""\s{2,}"""), " ")
        return clean.ifEmpty { name }
    }

    fun hasCover(name: String): Boolean = find(name) != null

    // The bundled cover of a known game, or null.
    fun cover(name: String): Bitmap? {
        val entry = find(name) ?: return null
        return try {
            YuzuApplication.appContext.assets.open("novas-covers/${entry.id}.webp").use { BitmapFactory.decodeStream(it) }
        } catch (e: Exception) {
            null
        }
    }
}
