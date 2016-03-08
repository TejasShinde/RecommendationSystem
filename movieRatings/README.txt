
This data set consists of:
	* 80,000 ratings (1-5) from 943 users on 1682 movies.
	* Data about each movie (title, release data, IMDb URL, genre info)
    * Simple demographic info for the users (age, gender, occupation, zip)


DETAILED DESCRIPTIONS OF DATA FILES
==============================================

Here are brief descriptions of the data.


ratings    -- 80000 ratings by 943 users on 1682 items.
              The data is organized as a 943x1682 matrix with entry (i,j)
              containing the rating from user i of movie j.


movieData  -- A cell array containing information about the items (movies);
              each row contains data for a single movie, the columns of this
              array contain:
              movie id | movie title | release date | IMDb URL |
              unknown | Action | Adventure | Animation | Children's | Comedy |
              Crime | Documentary | Drama | Fantasy | Film-Noir | Horror |
              Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |
              The last 19 columns (Unknown-Western) are the genres, a 1
              indicates the movie is of that genre, a 0 indicates it is not;
              note that movies can be in several genres at once
              The "movie id" corresponds to columns of the ratings matrix.

userData   -- A cell array containing demographic information about the users;
              each row contains data for a single user, the columns of this array
              contain:
              user id | age | gender | occupation | zip code
              The "user id" corresponds to rows of the ratings matrix.
